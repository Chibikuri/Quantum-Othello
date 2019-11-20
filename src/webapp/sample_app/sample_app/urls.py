from django.contrib import admin
from django.urls import include, path
from django.conf.urls.static import static
from django.conf import settings
from django.conf.urls import url

import manager.views as manager_view

urlpatterns = [
    url('admin/', admin.site.urls),
    url('home', manager_view.home, name='home'),
    url('quantum_othello', manager_view.try_quantum, name='q_try'),
    url('members', manager_view.members, name='members'),
    url('technologies', manager_view.technologies, name='tech'),
    url('links', manager_view.links, name='links'),
    path("ajax", manager_view.test_ajax_app),
    url('research', manager_view.detail, name='research'),
    url('4playboard', manager_view.four_board, name='four_board'),
    url('9playboard', manager_view.nine_board, name='nine_board'),
    url('16playboard', manager_view.sixteen_board, name='sixteen_board'),
]

#urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
