import csv
import random

# Define some sample names, middle names, and address components
first_names = [
    "John", "Mary", "Adam", "Sarah", "Michael", "Jane", "Maria",
    "Dave", "Anna", "Chris", "Lisa", "Marcus", "Karen", "Rob", "Diana"
]
middle_names = ["A.", "B.", "C.", "D.", "E.", "F.", "G.", "H.", "J.", "K."]
last_names = [
    "Smith", "Johnson", "Brown", "Clark", "Davis", "Williams", "Miller",
    "Wilson", "Taylor", "Anderson", "Martinez", "Lee", "Kim", "Allen", "Wright"
]
street_names = [
    "Redwood St", "Apple Dr", "Elm Street", "Maple Rd", "Oak Ave", "Pine Ln",
    "Birch Dr", "Cedar Ct", "Valley Rd", "Sunset Blvd"
]
cities = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia",
    "San Antonio", "San Diego", "Dallas", "San Jose"
]
states = ["NY", "CA", "TX", "IL", "AZ", "PA", "FL", "NC", "GA", "MI"]


def generate_address():
    # 10% chance the address is missing
    if random.random() < 0.1:
        return ""
    street_num = random.randint(1, 9999)
    street = random.choice(street_names)
    city = random.choice(cities)
    state = random.choice(states)
    zip_code = random.randint(10000, 99999)
    return f"{street_num} {street}, {city}, {state} {zip_code}"


def generate_row(case_type):
    """
    Generates one row based on the case type:
      Case 1: Exact match (same name and address) -> label 1
      Case 2: Same first & last name, different address -> label 0
      Case 3: Same first & last name but contributor gets a middle name and different address -> label 0
      Case 4: Same name but contributor's address is missing -> label 0
      Case 5: Partial address variation (e.g., slight apt addition or city change)
               - If apt added, label 1; if major change (city replaced), label 0.
    """
    # Generate employee name and address
    emp_first = random.choice(first_names)
    emp_last = random.choice(last_names)
    emp_address = generate_address()

    if case_type == 1:
        # Exact match: contributor same as employee
        ctr_first = emp_first
        ctr_last = emp_last
        ctr_address = emp_address
        label = 1
    elif case_type == 2:
        # Same name, but contributor gets a completely different address.
        ctr_first = emp_first
        ctr_last = emp_last
        ctr_address = generate_address()
        # Ensure the address is different
        while ctr_address == emp_address:
            ctr_address = generate_address()
        label = 0
    elif case_type == 3:
        # Same first and last, but contributor gets an extra middle name and different address.
        ctr_first = emp_first + " " + random.choice(middle_names)
        ctr_last = emp_last
        ctr_address = generate_address()
        while ctr_address == emp_address:
            ctr_address = generate_address()
        label = 0
    elif case_type == 4:
        # Same name, but contributor's address is missing.
        ctr_first = emp_first
        ctr_last = emp_last
        ctr_address = ""
        label = 0
    elif case_type == 5:
        # Partial address variation:
        # In 50% of these cases, the contributor gets an "Apt" appended -> label 1.
        # In the other 50%, change one component (e.g., city) -> label 0.
        ctr_first = emp_first
        ctr_last = emp_last
        if emp_address:
            if random.random() < 0.5:
                # Append an apartment number.
                ctr_address = emp_address + " Apt " + str(random.randint(1, 50))
                label = 1
            else:
                # Change city component.
                parts = emp_address.split(",")
                if len(parts) >= 3:
                    # Replace the city with a different random city.
                    parts[1] = " " + random.choice(cities)
                    ctr_address = ",".join(parts)
                    label = 0
                else:
                    ctr_address = emp_address
                    label = 1
        else:
            ctr_address = ""
            label = 0
    else:
        # Fallback: 50% chance match, 50% chance non-match.
        if random.random() < 0.5:
            ctr_first = emp_first
            ctr_last = emp_last
            ctr_address = emp_address
            label = 1
        else:
            ctr_first = emp_first
            ctr_last = emp_last
            ctr_address = generate_address()
            while ctr_address == emp_address:
                ctr_address = generate_address()
            label = 0

    return [emp_first, emp_last, emp_address, ctr_first, ctr_last, ctr_address, label]


def main():
    num_rows = 1000
    headers = [
        "employee_first_name", "employee_last_name", "employee_address",
        "contributor_first_name", "contributor_last_name", "contributor_address",
        "label"
    ]
    # We'll use 5 different cases; choose a case type at random for each row.
    case_types = [1, 2, 3, 4, 5]
    rows = []

    for i in range(num_rows):
        case = random.choice(case_types)
        row = generate_row(case)
        rows.append(row)

    output_file = "training_data_1000.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Generated {num_rows} rows in {output_file}")


if __name__ == "__main__":
    main()
