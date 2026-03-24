import random
from datetime import datetime, timedelta

class DatasetBuilder:
    def __init__(self):
        self.__filePath = "src/app/static/dataset.csv"

        # ORIGINAL gender list (kept)
        self.genders = ["Non binary", "Male", "Trans Female", "Female", "Trans Male"]
        self.gender_weights = [0.05, 0.36, 0.10, 0.40, 0.09]

        # ORIGINAL ethnicity list (kept)
        self.ethnicities = [
            "Asian/British Asian - Chinese", "Asian/British Asian - Other",
            "Black/Black British - Other", "Gypsy / Roma", "Black/Black British - African",
            "White - British", "White - Irish", "Asian/British Asian - Indian",
            "White - Other", "Black/Black British - Caribbean",
            "Asian/British Asian - Pakistani", "Asian/British Asian - Bangladeshi",
            "Mixed - White/Black African", "Traveller of Irish Heritage",
            "Mixed - White/Asian", "Mixed - Other", "Traveller - Other",
            "White - Central European", "Mixed - White/Black Caribbean",
            "Dual Heritage - Black/White", "White - Eastern European"
        ]

        # ORIGINAL ethnicity weights (kept)
        self.ethnicity_weights = [
            0.002, 0.015, 0.002, 0.005, 0.013, 0.567, 0.004, 0.026,
            0.036, 0.005, 0.255, 0.023, 0.002, 0.002, 0.012, 0.005,
            0.005, 0.036, 0.008, 0.015, 0.036
        ]

        # BCFT-refined placement types
        self.placement_types = [
            "Kinship",
            "External Fostering",
            "In-House Fostering",
            "Residential"
        ]

        self.placement_weights = [
            0.2122,   # Kinship
            0.3364,   # External Fostering
            0.4514,   # In-House Fostering
            0.1015    # Residential
        ]

        # BCFT placement-days distribution
        self.length_buckets = [
            (1, 100),
            (101, 300),
            (301, 500),
            (501, 900),
            (901, 1500),
            (1501, 3000),
            (3001, 4000)
        ]
        self.length_weights = [0.439, 0.227, 0.173, 0.145, 0.064, 0.043, 0.012]

        # Move reasons (simplified + realistic)
        self.move_reasons = [
            "Planned move",
            "Carer requests end",
            "Child requests end",
            "Behaviour concerns",
            "Allegation",
            "Placement stability concerns"
        ]

    def _weighted_choice(self, options, weights):
        return random.choices(options, weights=weights, k=1)[0]

    def _random_date(self):
        start = datetime.strptime("2022-01-01", "%Y-%m-%d")
        end = datetime.strptime("2025-10-31", "%Y-%m-%d")
        delta = (end - start).days
        return start + timedelta(days=random.randint(0, delta))

    def create_dataset(self, num_rows=1000):
        data = []

        for _ in range(num_rows):

            # Child attributes
            child_age = random.choices(
                population=list(range(0, 18)),
                weights=[0.02,0.02,0.03,0.03,0.05,0.07,0.10,0.12,0.13,0.13,0.10,0.08,0.05,0.04,0.02,0.01,0.01,0.01],
                k=1
            )[0]

            child_gender = self._weighted_choice(self.genders, self.gender_weights)
            child_ethnicity = self._weighted_choice(self.ethnicities, self.ethnicity_weights)

            prior_placements = random.randint(0, 4)
            returning_child = random.choice([True, False])
            missing_episodes = random.randint(0, 7)

            # Siblings
            sibling_group = random.randint(0, 5)
            placed_with_siblings = sibling_group > 1

            # Carer attributes
            carer_age = random.randint(25, 75)
            carer_gender = self._weighted_choice(self.genders, self.gender_weights)
            carer_ethnicity = self._weighted_choice(self.ethnicities, self.ethnicity_weights)

            # Placement type
            placement_type = self._weighted_choice(self.placement_types, self.placement_weights)

            # Distance from home
            if placement_type == "Kinship":
                distance = round(random.uniform(0, 5), 2)
            elif placement_type in ["In-House Fostering", "External Fostering"]:
                distance = round(random.uniform(0, 20), 2)
            else:  # Residential
                distance = round(random.uniform(10, 100), 2)

            # Placement duration
            bucket = self._weighted_choice(self.length_buckets, self.length_weights)
            days_placed = random.randint(bucket[0], bucket[1])

            # Dates
            start_date = self._random_date()
            move_date = start_date + timedelta(days=days_placed)

            # Move reason
            move_reason = random.choice(self.move_reasons)

            # Additional contextual fields
            eh_involvement = random.choice([True, False])
            yot_involvement = random.choice([True, False])

            row = {
                "Child Age At Placement": child_age,
                "Child Gender": child_gender,
                "Child Ethnicity": child_ethnicity,
                "Child Prior Placements": prior_placements,
                "Returning Child": returning_child,
                "Missing Episodes": missing_episodes,
                "Sibling Group Size": sibling_group,
                "Placed With Siblings": placed_with_siblings,

                "Carer Age": carer_age,
                "Carer Gender": carer_gender,
                "Carer Ethnicity": carer_ethnicity,

                "Placement Type": placement_type,
                "Placement Start Date": start_date.strftime("%Y-%m-%d"),
                "Move Date": move_date.strftime("%Y-%m-%d"),
                "Days Placed": days_placed,
                "Move Reason": move_reason,
                "Distance From Home (miles)": distance,

                "EH involvement": eh_involvement,
                "YOT involvement": yot_involvement,

                "Placement Sequence Number": prior_placements + 1
            }

            data.append(row)

        return data

    def write_dataset(self, data):
        with open(self.__filePath, 'w') as file:
            headers = list(data[0].keys())
            file.write(",".join(headers) + "\n")
            for row in data:
                file.write(",".join(str(row[h]) for h in headers) + "\n")

    def get_file_path(self):
        return self.__filePath


if __name__ == "__main__":
    dsb = DatasetBuilder()
    rows = dsb.create_dataset(3000)
    dsb.write_dataset(rows)
    print(f"Generated {len(rows)} rows at {dsb.get_file_path()}. Example:")
    print(rows[random.randint(0,2999)])
