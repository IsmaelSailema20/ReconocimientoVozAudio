from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('logout/', views.logout_view, name='logout'),
    path('api/recognize/', views.recognize_face, name='recognize_face'),
    path('api/verify/', views.verify_face_stream, name='verify_face'),
    path('api/check-session/', views.check_session, name='check_session'),
]
