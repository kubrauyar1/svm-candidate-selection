# Faker Library Documentation

## What is Faker?

Faker is a Python library for generating fake data. It is used to create realistic-looking but completely random data for testing, demo applications, and data anonymization.

## Key Features

1. **Multi-language Support**: Can generate data in over 20 languages
2. **Rich Data Types**: Names, addresses, phone numbers, emails, company information, etc.
3. **Customizable**: You can add your own providers
4. **Localized Data**: Language-specific data formats for each supported language

## Installation

```bash
pip install Faker
```

## Basic Usage

```python
from faker import Faker

# Create Faker object
fake = Faker()

# For Turkish data
fake_tr = Faker('tr_TR')

# Basic data generation examples
print(fake.name())          # Random name
print(fake.address())       # Random address
print(fake.email())         # Random email
print(fake.company())       # Random company name
print(fake.job())           # Random job title
print(fake.phone_number())  # Random phone number
```

## Example Usage for Job Application Project

```python
from faker import Faker
import numpy as np

fake = Faker('tr_TR')

def generate_applicant_data(n_samples):
    data = []
    for _ in range(n_samples):
        applicant = {
            'full_name': fake.name(),
            'experience_years': np.random.uniform(0, 10),
            'technical_score': np.random.uniform(0, 100),
            'university': fake.company() + ' University',
            'department': fake.job() + ' Engineering',
            'email': fake.email(),
            'phone': fake.phone_number()
        }
        data.append(applicant)
    return data

# Generate data for 10 applicants
applicants = generate_applicant_data(10)
for applicant in applicants:
    print(applicant)
```

## Creating Custom Providers

```python
from faker import Faker
from faker.providers import BaseProvider

class CustomProvider(BaseProvider):
    def programming_language(self):
        languages = ['Python', 'Java', 'JavaScript', 'C++', 'C#', 'Ruby', 'Go']
        return self.random_element(languages)

fake = Faker()
fake.add_provider(CustomProvider)

print(fake.programming_language())  # Random programming language
```

## Creating and Saving Dataset to CSV

```python
import pandas as pd
from faker import Faker

fake = Faker('tr_TR')

def generate_dataset(n_samples):
    data = []
    for _ in range(n_samples):
        row = {
            'full_name': fake.name(),
            'experience_years': np.random.uniform(0, 10),
            'technical_score': np.random.uniform(0, 100),
            'university': fake.company() + ' University',
            'department': fake.job() + ' Engineering',
            'email': fake.email(),
            'phone': fake.phone_number(),
            'programming_language': fake.programming_language()
        }
        data.append(row)
    return pd.DataFrame(data)

# Generate dataset for 100 applicants
df = generate_dataset(100)

# Save to CSV
df.to_csv('applicant_data.csv', index=False)
```

## Advantages of Faker

1. **Realistic Data**: Generated data resembles real-world data
2. **Fast Data Generation**: Large datasets can be created quickly
3. **Testing Ease**: Ideal for test scenarios
4. **Privacy**: Can be used instead of real data
5. **Customizability**: Can be customized according to needs

## Limitations

1. Data is completely random, so it may not capture correlations present in real datasets
2. Maintaining data consistency can be challenging in some cases
3. Custom providers may be needed for very specific data types

## Tips

1. Use seed values for reproducibility when generating data
2. Create your own providers for specific needs
3. Use generator pattern for large datasets
4. Generate related data together for consistency 