# Generated by Django 5.1.1 on 2024-09-18 16:27

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('movie', '0006_alter_movie_image'),
    ]

    operations = [
        migrations.AlterField(
            model_name='movie',
            name='image',
            field=models.ImageField(default='movie/images/default.jpg', upload_to='movie/images/'),
        ),
    ]