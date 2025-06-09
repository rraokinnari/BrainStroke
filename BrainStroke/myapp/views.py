from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.core.files.storage import FileSystemStorage
from .models import Doctor, Patient, CTScan, PredictionResult
from .predictor import predict_stroke
import cv2
import os
from django.conf import settings
from django.db import IntegrityError
from .predictor import scalar, model, class_indices
from .gradcam import generate_gradcam_overlay
from PIL import Image
import io
from django.core.files.base import ContentFile


# ---------------------- AUTH ----------------------

from django.http import HttpResponse

def register_doctor(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        name = request.POST['name']
        age = request.POST['age']
        gender = request.POST['gender']

        if User.objects.filter(username=username).exists():
            return render(request, 'register.html', {
                'error': 'Username already taken. Please choose another.'
            })

        try:
            user = User.objects.create_user(username=username, password=password)
            Doctor.objects.create(user=user, name=name, age=age, gender=gender)
            return redirect('login')
        except IntegrityError:
            return render(request, 'register.html', {
                'error': 'A user with this username already exists.'
            })
        except Exception as e:
            return HttpResponse(f"Error: {str(e)}")  # <== shows exact error

    return render(request, 'register.html')


def login_doctor(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)

        if user:
            login(request, user)
            return redirect('dashboard')
        else:
            return render(request, 'login.html', {'error': 'Invalid credentials'})

    return render(request, 'login.html')


def logout_doctor(request):
    logout(request)
    return redirect('login')


# ---------------------- DASHBOARD ----------------------

@login_required
def dashboard(request):
    doctor = Doctor.objects.get(user=request.user)
    patients = Patient.objects.filter(doctor=doctor)
    return render(request, 'dashboard.html', {'doctor': doctor, 'patients': patients})


@login_required
def add_patient(request):
    if request.method == 'POST':
        name = request.POST['name']
        age = request.POST['age']
        gender = request.POST['gender']
        doctor = Doctor.objects.get(user=request.user)

        Patient.objects.create(name=name, age=age, gender=gender, doctor=doctor)
        return redirect('dashboard')

    return render(request, 'add_patient.html')


# ---------------------- CT SCAN UPLOAD ----------------------

from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.core.files.storage import FileSystemStorage
from django.conf import settings

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from .models import Doctor, Patient, CTScan, PredictionResult
from .predictor import model, class_indices, scalar  # Make sure your model & scalar() are here

# Grad-CAM helper functions
def load_img_for_gradcam(img_path):
    img = image.load_img(img_path, target_size=(224, 224), color_mode='rgb')
    img_array = image.img_to_array(img)
    img_array = scalar(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, np.array(img) / 255.0

def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_output = predictions[:, pred_index]

    grads = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def apply_gradcam(img_path, model, scalar_func, class_indices, last_conv_layer_name='block5_conv3'):
    img_array, raw_img = load_img_for_gradcam(img_path)
    heatmap = get_gradcam_heatmap(model, img_array, last_conv_layer_name)

    heatmap = cv2.resize(heatmap, (raw_img.shape[1], raw_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap_color * 0.4 + raw_img * 255
    superimposed_img = np.uint8(superimposed_img)

    preds = model.predict(img_array)
    class_labels = {v: k for k, v in class_indices.items()}
    pred_class = class_labels[np.argmax(preds)]
    confidence = 100 * np.max(preds)

    return superimposed_img, pred_class, confidence


@login_required
def upload_ct_scan(request):
    doctor = Doctor.objects.get(user=request.user)
    patients = Patient.objects.filter(doctor=doctor)

    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        patient_id = request.POST.get('patient_id')

        try:
            patient = Patient.objects.get(id=patient_id, doctor=doctor)
        except Patient.DoesNotExist:
            return render(request, 'upload.html', {
                'error': 'Invalid patient selected',
                'patients': patients
            })

        # Save the uploaded image
        fs = FileSystemStorage()
        filename = fs.save(image_file.name, image_file)
        file_path = fs.path(filename)

        # Predict using model
        from tensorflow.keras.preprocessing import image
        import numpy as np
        img = image.load_img(file_path, target_size=(224, 224), color_mode='rgb')
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        preds = model.predict(img_array)
        pred_class_idx = np.argmax(preds)
        confidence = float(np.max(preds)) * 100
        label = [k for k, v in class_indices.items() if v == pred_class_idx][0]

        # Save CT Scan object
        ct_scan = CTScan.objects.create(
            patient=patient,
            image=filename,
            prediction=label,
            doctor=request.user
        )

        # Save prediction result
        PredictionResult.objects.create(
            ct_scan=ct_scan,
            label=label,
            confidence_score=confidence
        )

        # Generate Grad-CAM visualization
        gradcam_img, _, _ = generate_gradcam_overlay(
            img_path=file_path,
            model=model,
            class_indices=class_indices,
            last_conv_layer_name='block5_conv3'  # ✅ Change if your model differs
        )

        # Save Grad-CAM overlay image
        gradcam_filename = f'gradcam_{filename}'
        gradcam_path = os.path.join(settings.MEDIA_ROOT, gradcam_filename)
        cv2.imwrite(gradcam_path, gradcam_img)

        return render(request, 'result.html', {
            'ct_scan': ct_scan,
            'confidence': confidence,
            'gradcam_image': f'media/{gradcam_filename}'
        })

    return render(request, 'upload.html', {'patients': patients})

# ---------------------- PATIENT HISTORY ----------------------

@login_required
def patient_history(request, patient_id):
    doctor = Doctor.objects.get(user=request.user)
    try:
        patient = Patient.objects.get(id=patient_id, doctor=doctor)
    except Patient.DoesNotExist:
        return redirect('dashboard')

    scans = CTScan.objects.filter(patient=patient).order_by('-uploaded_at')
    return render(request, 'history.html', {'patient': patient, 'scans': scans})
