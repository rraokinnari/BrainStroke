from django.contrib import admin
from .models import Doctor, Patient, CTScan, PredictionResult

admin.site.register(Doctor)
admin.site.register(Patient)
admin.site.register(CTScan)
admin.site.register(PredictionResult)
