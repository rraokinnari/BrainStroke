from django.contrib import admin
from .models import Doctor, AdminStaff, Patient, CTScan, PredictionResult

admin.site.register(Doctor)
admin.site.register(AdminStaff)
admin.site.register(Patient)
admin.site.register(CTScan)
admin.site.register(PredictionResult)
