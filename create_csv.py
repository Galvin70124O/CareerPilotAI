import csv

# Data for Resume.csv
rows = [
    ["Resume", "Category"],
    ["Python programming, machine learning, data analysis", "Data Science"],
    ["Web development, HTML, CSS, JavaScript", "Web Developer"],
    ["Database management, SQL, Oracle", "Database Administrator"],
    ["Java development, object oriented programming", "Java Developer"],
    ["C programming, embedded systems, electronics", "Embedded Engineer"],
    ["Communication, negotiation, team management", "HR Manager"]
]

# Create or overwrite Resume.csv
with open('Resume.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(rows)

print("Resume.csv created successfully!")
