import random
from datetime import datetime, timedelta

class DatasetBuilder:
    def __init__(self):
        self.__filePath = "dataset.csv"
        genders = ["Non binary", "Indeterminate", "Unknown", "NULL", "Male",
                   "Unborn", "Trans Female", "Female", "Unspecified", "Trans Male"]
        ethnicities = ["Information Not Yet Obtained", "Asian/British Asian – Chinese",
                       "Asian/British Asian – Other", "Black/Black British – Other",
                       "Gypsy / Roma", "Black/Black British – African", "White – British",
                       "White – Irish", "Asian/British Asian – Indian", "Unknown",
                       "White – Other", "Black/Black British – Caribbean", "NULL",
                       "Asian/British Asian – Pakistani", "Asian/British Asian – Bangladeshi",
                       "Mixed - White/Black African", "Traveller of Irish Heritage",
                       "Mixed - White/Asian", "Refused", "Mixed – Other", "Traveller – Other",
                       "Gypsy", "Roma", "White - Central European", "Mixed - White/Black Caribbean",
                       "Other Ethnic Group", "Dual Heritage - Black/White", "White - Eastern European"]

        # UPDATED PLACEMENT TYPE WEIGHTINGS BASED ON REAL DATA
        placement_types = [
            "Fostering - Long Term",
            "Fostering - Short Term",
            "Kinship",
            "Residential",
            "Special Guardianship",
            "Fostering - Emergency",
            "Fostering - Respite",
            "Adoption"
        ]

        placement_weights = [
            0.045,   # Fostering - Long Term
            0.204,   # Fostering - Short Term
            0.576,   # Kinship
            0.070,   # Residential
            0.050,   # Special Guardianship
            0.013,   # Fostering - Emergency
            0.002,   # Fostering - Respite
            0.039    # Adoption
        ]

        self.__fields = {
            "Child Age At Placement": list(range(0, 18)),
            "Child Gender": genders,
            "Child Ethnicity": ethnicities,
            "Child Prior Placements Number": [0],
            "Returning Child": [True, False],
            "Age Child Left Care": list(range(1, 19)),
            "Carer Age": [0, 0],
            "Placement Type": placement_types,
            "Missing Episodes": list(range(0, 8)),
            "Placement Sequence Number": [0],
            "Placement Start Date": datetime.now(),
            "Placement End Reason": [
                "Allegation (S47)",
                "Approval removed",
                "Carer requests placement end due to child's behaviour",
                "Child requests placement end",
                "Planned move to long-term fostering"
            ],
            "Placement Time Period (days)": [0],
            "Number Of Carers": [0],
            "Carer Gender Composition": [genders, genders],
            "Placed With Siblings": [True, False],
            "Emergency Placement": [True, False],
            "Distance From Home (miles)": [0.0],
            "Placement Planning Meeting": [True, False],
            "Reason For Leaving Care": "Why the child left care entirely",
            "Sibling Group Size": list(range(0, 9)),
            "Previous Care History": "Any additional context and comments about the child’s care history",
            "Age Child Came Into Care": [0],
            "Residential Home Type": "including Mainstream",
            "Carer Ethnicity Or Religion": [ethnicities, ethnicities],
            "Carer Type": [
                "Long Term Foster", "Short Term Foster", "Kinship", "Residential",
                "Special Guardianship", "Emergency Foster", "Respite Foster", "Adoption"
            ],
            "EH involvement": [True, False],
            "Siblings In EH": [True, False],
            "YOT involvement": [True, False]
        }

        self.__weighting = ([
            "Eq", [0.05, 0, 0, 0, 0.36, 0, 0.1, 0.4, 0, 0.09],
            [[0, 0.002, 0.015, 0.002, 0.005, 0.013, 0.567, 0.004, 0.026, 0,
             0.036, 0.005, 0, 0.255, 0.023, 0.002, 0.002, 0.012, 0, 0.005,
             0.005, 0.002, 0.003, 0.036, 0.008, 0.015, 0.005, 0.036], "Max 5"],
            [0.55, 0.30, 0.10, 0.03, 0.02], [0.6, 0.4], "Eq", "Min 25, Max 75", placement_weights,
            [0, 0.5, 0.25, 0.12, 0.06, 0.03, 0.02, 0.01, 0.01], "Min 1, Max 1000", "Max 5",
            [[0.05, 0, 0, 0, 0.36, 0, 0.1, 0.4, 0, 0.09], "Max 5"], "Eq", "Eq", "Eq", "Eq", "Eq",
            [0.6, 0, 0.2, 0.1, 0.05, 0.03, 0.01, 0.01], "Eq", "Max 17", "Eq",
            [[0, 0.002, 0.015, 0.002, 0.005, 0.013, 0.567, 0.004, 0.026, 0,
              0.036, 0.005, 0, 0.255, 0.023, 0.002, 0.002, 0.012, 0, 0.005,
              0.005, 0.002, 0.003, 0.036, 0.008, 0.015, 0.005, 0.036], "Max 5"],
            "Eq", [0.3, 0.7], [0.25, 0.75], [0.1, 0.9]
        ])

    def _resolve_weight_list(self, weighting):
        if isinstance(weighting, list):
            for item in weighting:
                if isinstance(item, list) and all(isinstance(x, (int, float)) for x in item):
                    return item
            if all(isinstance(x, (int, float)) for x in weighting):
                return weighting
        return None

    def _column_values(self, options, weighting, n):
        if not isinstance(options, list):
            return [options] * n

        if isinstance(weighting, str) and "Min" in weighting and "Max" in weighting:
            parts = weighting.replace("Min", "").replace("Max", "").split(",")
            min_val = float(parts[0].strip())
            max_val = float(parts[1].strip())
            return [random.uniform(min_val, max_val) for _ in range(n)]

        if isinstance(weighting, str) and "Max" in weighting and "Min" not in weighting:
            max_val = int(weighting.replace("Max", "").strip())
            return [random.randint(0, max_val) for _ in range(n)]

        weights = self._resolve_weight_list(weighting)
        if weights and len(weights) == len(options) and sum(weights) > 0:
            return random.choices(options, weights=weights, k=n)

        if weighting == "Eq" or weights is None:
            return [random.choice(options) for _ in range(n)]

        return [random.choice(options) for _ in range(n)]

    def create_dataset(self, num_rows=1000):
        headers = list(self.__fields.keys())
        columns = {}

        for idx, key in enumerate(headers):
            options = self.__fields[key]
            weighting = self.__weighting[idx] if idx < len(self.__weighting) else None
            columns[key] = self._column_values(options, weighting, num_rows)

        for i in range(num_rows):
            placement_age = columns["Child Age At Placement"][i]
            columns["Age Child Left Care"][i] = random.randint(placement_age + 1, 18)

            if not columns["Returning Child"][i]:
                columns["Placement End Reason"][i] = "N/A"
                columns["Placement Time Period (days)"][i] = "N/A"
                columns["Reason For Leaving Care"][i] = "N/A"
                columns["Previous Care History"][i] = "N/A"

            columns["Placement Sequence Number"][i] = columns["Child Prior Placements Number"][i] + 1

            placement_to_carer = {
                "Fostering - Long Term": "Long Term Foster",
                "Fostering - Short Term": "Short Term Foster",
                "Kinship": "Kinship",
                "Residential": "Residential",
                "Special Guardianship": "Special Guardianship",
                "Fostering - Emergency": "Emergency Foster",
                "Fostering - Respite": "Respite Foster",
                "Adoption": "Adoption"
            }
            columns["Carer Type"][i] = placement_to_carer[columns["Placement Type"][i]]

            num_carers = max(1, columns["Number Of Carers"][i])
            columns["Number Of Carers"][i] = num_carers

            ptype = columns["Placement Type"][i]

            if ptype == "Kinship":
                distance = random.uniform(0, 5)
            elif ptype in ["Fostering - Long Term", "Fostering - Short Term",
                           "Fostering - Emergency", "Fostering - Respite"]:
                distance = random.uniform(0, 20)
            elif ptype == "Residential":
                distance = random.uniform(10, 100)
            elif ptype == "Special Guardianship":
                distance = random.uniform(0, 15)
            elif ptype == "Adoption":
                distance = random.uniform(0, 25)
            else:
                distance = random.uniform(0, 20)

            columns["Distance From Home (miles)"][i] = round(distance, 2)

            columns["Carer Age"][i] = [random.randint(25, 75) for _ in range(num_carers)]

            genders = self.__fields["Child Gender"]
            columns["Carer Gender Composition"][i] = [random.choice(genders) for _ in range(num_carers)]

            ethnicities = self.__fields["Child Ethnicity"]
            columns["Carer Ethnicity Or Religion"][i] = [random.choice(ethnicities) for _ in range(num_carers)]

            sibling_size = columns["Sibling Group Size"][i]
            if sibling_size > 1:
                columns["Placed With Siblings"][i] = True
            if sibling_size < 2:
                columns["Siblings In EH"][i] = False

            length_days = random.choices(
                [random.randint(1, 7), random.randint(30, 200), random.randint(200, 1000)],
                weights=[0.1, 0.4, 0.5]
            )[0]

            columns["Placement Time Period (days)"][i] = length_days

            reason = columns["Placement End Reason"][i]

            # Allegation → kinship or short-term fostering
            if reason == "Allegation (S47)":
                if columns["Placement Type"][i] not in ["Fostering - Short Term", "Kinship"]:
                    columns["Placement Type"][i] = random.choice(["Fostering - Short Term", "Kinship"])

            # Approval removed → short-term fostering
            if reason == "Approval removed":
                columns["Placement Type"][i] = "Fostering - Short Term"

            # Behaviour-related endings → kinship or short-term fostering
            if "behaviour" in reason.lower():
                columns["Placement Type"][i] = random.choice(["Kinship", "Fostering - Short Term"])

            # Generate random placement start date between 01/01/2022 and 31/10/2025
            start_date = datetime.strptime("2022-01-01", "%Y-%m-%d")
            end_date = datetime.strptime("2025-10-31", "%Y-%m-%d")

            # Calculate random offset in days
            days_between = (end_date - start_date).days
            random_offset = random.randint(0, days_between)

            # Assign the generated date
            columns["Placement Start Date"][i] = start_date + timedelta(days=random_offset)

        data = [{key: columns[key][i] for key in headers} for i in range(num_rows)]
        return data

    def write_dataset(self, data):
        with open(self.__filePath, 'w') as file:
            headers = list(self.__fields.keys())
            file.write(",".join(headers) + "\n")
            for row in data:
                file.write(",".join(str(value) for value in row.values()) + "\n")

    def get_fields(self):
        return self.__fields

    def get_weights(self):
        return self.__weighting

    def get_file_path(self):
        return self.__filePath

    def set_file_path(self, new_file_path):
        self.__filePath = new_file_path

if __name__ == "__main__":
    dsb = DatasetBuilder()
    rows = dsb.create_dataset(3000)
    dsb.write_dataset(rows)
    print(f"Rows generated: {len(rows)}; First row below")
    for field in dsb.get_fields().keys():
        print(f"{field}: {rows[1][field]}")
