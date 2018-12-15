from django.urls import path
from nepaliocr import views
urlpatterns = [
    
    path('',views.index),
]