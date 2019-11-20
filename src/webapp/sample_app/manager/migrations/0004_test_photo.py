from django.db import migrations, models
 class Migration(migrations.Migration):
     dependencies = [
        ('manager', '0003_auto_20181126_1313'),
    ]
     operations = [
        migrations.CreateModel(
            name='test_Photo',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to='test_data')),
            ],
        ),
    ]
