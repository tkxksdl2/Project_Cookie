from django.urls import path

from animateapp.views import AnimateCreateView, AnimateDetailView

app_name = 'animateapp'

urlpatterns = [
    path('create/', AnimateCreateView.as_view(), name='create'),
    path('detail/<int:pk>', AnimateDetailView.as_view(), name='detail'),
]