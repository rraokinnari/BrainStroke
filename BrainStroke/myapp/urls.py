from django.urls import path
from . import views
from django.urls import path,include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    # Authentication
    path('', views.login_doctor, name='login'),
    path('register/', views.register_doctor, name='register'),
    path('logout/', views.logout_doctor, name='logout'),

    # Dashboard
    path('dashboard/', views.dashboard, name='dashboard'),
    path('add_patient/', views.add_patient, name='add_patient'),

    # CT Scan Upload and Result
    path('upload/', views.upload_ct_scan, name='upload'),

    # Patient History
    path('history/<int:patient_id>/', views.patient_history, name='history'),
]+static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)
