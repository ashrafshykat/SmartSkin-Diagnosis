from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings

from analysis import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('analysis.urls')),
    path('upload/', views.upload, name='upload'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
