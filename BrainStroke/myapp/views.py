from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.contrib.auth.decorators import login_required
from .models import CTScan, Patient, PredictionResult
from .predictor import predict_stroke

@login_required
def upload_ct_scan(request):
    patients = Patient.objects.all()  # Available for both GET and error cases

    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        patient_id = request.POST.get('patient_id')

        try:
            patient = Patient.objects.get(id=patient_id)
        except Patient.DoesNotExist:
            return render(request, 'upload.html', {
                'error': 'Invalid patient selected',
                'patients': patients
            })

        fs = FileSystemStorage()
        filename = fs.save(image_file.name, image_file)
        file_path = fs.path(filename)

        # Predict using model
        label, confidence = predict_stroke(file_path)

        # Create CTScan entry
        ct_scan = CTScan.objects.create(
            patient=patient,
            image=filename,
            prediction=label,
            doctor=request.user
        )

        # Create detailed result
        PredictionResult.objects.create(
            ct_scan=ct_scan,
            label=label,
            confidence_score=confidence
        )

        return render(request, 'result.html', {
            'ct_scan': ct_scan,
            'confidence': confidence
        })

    return render(request, 'upload.html', {'patients': patients})
