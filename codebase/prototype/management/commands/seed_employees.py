from django.core.management.base import BaseCommand
from faker import Faker
import random

from prototype.models import Organization, Contributions
from prototype.models.contributions import Contributions


class Command(BaseCommand):
    help = 'Seeds the database with random Employee data.'

    def handle(self, *args, **options):
        fake = Faker()
        name_list = [
            ("Maria" ,"Calas"),
            ("Maria", "J.Calas"),
            ("Maria", "M. Calas"),
            ("Maria", "Ujku Calas"),
            ("Maria", "Test Calas")
        ]

        # Ensure there's at least one organization
        orgs = Organization.objects.all()
        if not orgs.exists():
            self.stdout.write(self.style.ERROR("No organizations found. Please create at least one."))
            return

        # Let's create 10 employees
        for name in name_list:
            email = f"{name[0].lower()}.{name[1].replace(' ', '').lower()}@example.com"

            org = random.choice(orgs)
            employee = Contributions.objects.create(
                first_name=name[0],
                last_name=name[1],
                email=email,
                amount=fake.random_int(min=1, max=1000),
                recipient=f"{fake.first_name()} {fake.last_name()}",
            )
            self.stdout.write(f"Created Contributions: {employee}")

        self.stdout.write(self.style.SUCCESS("Successfully seeded Contributions data!"))
