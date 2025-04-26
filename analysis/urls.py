from django.urls import path
from .views import index, upload, realtime, analyze

urlpatterns = [
    path('', index, name='home'),
    path('upload/', upload, name='upload'),
    path('analyze/', analyze, name='analyze'),
    path('realtime/', realtime, name='realtime'),
]
