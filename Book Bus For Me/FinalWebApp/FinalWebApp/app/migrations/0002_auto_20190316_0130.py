# -*- coding: utf-8 -*-
# Generated by Django 1.11.20 on 2019-03-15 20:00
from __future__ import unicode_literals

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0001_initial'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='Bookings',
            new_name='Booking',
        ),
    ]
