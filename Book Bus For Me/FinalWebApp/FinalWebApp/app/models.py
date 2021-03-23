"""
Definition of models.
"""

from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User

class Booking(models.Model):
    username=models.ForeignKey(User,on_delete=models.CASCADE)
    fromPlace=models.CharField(max_length=25)
    toPlace=models.CharField(max_length=25)
    #departureDate=models.DateTimeField(default=timezone.now)
    #returnDate=models.DateTimeField(default=timezone.now)
    departureDate=models.CharField(max_length=20)
    returnDate=models.CharField(max_length=20)
    tripType=models.CharField(max_length=25, default="Round")

    def __str__(self):
        return self.toPlace

# Create your models here.
