import random
import csv
from datetime import datetime, timedelta
import us  # US States library

# Generate 100 unique part numbers
def generate_part_numbers(num_parts=100):
    parts = []
    for i in range(num_parts):
        prefix = random.choice(['A', 'B', 'C', 'D', 'E'])
        number = str(random.randint(1000, 9999))
        suffix = random.choice(['X', 'Y', 'Z'])
        part_number = f"{prefix}{number}{suffix}"
        parts.append(part_number)
    return parts

# Generate random dates within a range
def random_date(start_date, end_date):
    time_between = end_date - start_date
    days_between = time_between.days
    random_days = random.randrange(days_between)
    return start_date + timedelta(days=random_days)

# Generate the sales data
def generate_sales_data(num_records=10000):
    # Setup initial data
    part_numbers = generate_part_numbers()
    states = [state.name for state in us.states.STATES]
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    # Generate sales records
    sales_data = []
    for _ in range(num_records):
        date = random_date(start_date, end_date)
        part_number = random.choice(part_numbers)
        quantity = random.randint(1, 1000)
        unit_price = round(random.uniform(10.0, 500.0), 2)
        order_amount = round(quantity * unit_price, 2)
        state = random.choice(states)
        
        sales_data.append([
            date.strftime('%Y-%m-%d'),
            part_number,
            quantity,
            unit_price,
            order_amount,
            state
        ])
    
    return sales_data

# Generate and write the data to a CSV file
sales_data = generate_sales_data()
header = ['Date', 'Part_Number', 'Quantity', 'Unit_Price', 'Order_Amount', 'State']

with open('sales_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(sales_data)