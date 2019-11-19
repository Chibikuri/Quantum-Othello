from django.db import models

# Create your models here.
class Photo(models.Model):
    image = models.ImageField(upload_to='myapp')

#class test_Photo(models.Model):
#    image = models.ImageField(upload_to='test_data')
