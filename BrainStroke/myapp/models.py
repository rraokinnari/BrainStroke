from django.db import models
from django.contrib.auth.models import User

# User extension for Doctors
class Doctor(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    specialization = models.CharField(max_length=100)
    hospital_name = models.CharField(max_length=255)

    def __str__(self):
        return f"Dr. {self.user.get_full_name()} - {self.specialization}"

# AdminStaff profile (if you want to distinguish roles)
class AdminStaff(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    department = models.CharField(max_length=100)

    def __str__(self):
        return f"{self.user.get_full_name()} - Admin ({self.department})"

# Patient model
class Patient(models.Model):
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
    ]

    name = models.CharField(max_length=255)
    dob = models.DateField()
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES)
    medical_history = models.TextField(blank=True, null=True)
    contact_info = models.TextField(blank=True, null=True)

    def __str__(self):
        return self.name

# CTScan model
class CTScan(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='ct_scans/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    prediction = models.CharField(max_length=100, blank=True, null=True)
    doctor = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return f"{self.patient.name} - {self.prediction or 'Pending'}"

# Optional: Detailed prediction result
class PredictionResult(models.Model):
    ct_scan = models.OneToOneField(CTScan, on_delete=models.CASCADE)
    label = models.CharField(max_length=100)
    confidence_score = models.FloatField()
    prediction_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.label} ({self.confidence_score:.2f}) for {self.ct_scan.patient.name}"
