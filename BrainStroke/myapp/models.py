from django.db import models
from django.contrib.auth.models import User

# Doctor model (One-to-One with User for extended info)
class Doctor(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    age = models.IntegerField()
    gender = models.CharField(max_length=10)

    def __str__(self):
        return self.name

# Patient model (Each patient is associated with a doctor)
class Patient(models.Model):
    doctor = models.ForeignKey(Doctor, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    age = models.IntegerField()
    gender = models.CharField(max_length=10)

    def __str__(self):
        return self.name

# CT Scan model
class CTScan(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='ct_scans/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    prediction = models.CharField(max_length=100, blank=True, null=True)
    doctor = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return f"{self.patient.name} - {self.prediction or 'Pending'}"

# Prediction Result (stores confidence)
class PredictionResult(models.Model):
    ct_scan = models.OneToOneField(CTScan, on_delete=models.CASCADE)
    label = models.CharField(max_length=100)
    confidence_score = models.FloatField()

    def __str__(self):
        return f"{self.label} ({self.confidence_score:.2f}%)"
