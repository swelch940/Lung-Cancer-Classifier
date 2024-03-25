from django.db import models
import os

# Create your models here.
# models.py
class CTSCAN(models.Model):
    name = models.CharField(max_length=50)
    #file = models.FileField()
    ct_scan_Img = models.ImageField(upload_to='images/')
    
    #def filename(self):
        #return os.path.basename(self.file.name)