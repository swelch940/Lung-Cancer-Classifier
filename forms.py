from django import forms
from .models import CTSCAN
 
 
class CTForm(forms.ModelForm):
    class Meta:
        model = CTSCAN
        fields = ['name', 'ct_scan_Img']

