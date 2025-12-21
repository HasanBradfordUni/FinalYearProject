import random
from datetime import datetime

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
        self.__fields = {"Child Age At Placement":	[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],
                         "Child Gender": genders,
                         "Child Ethnicity":	ethnicities,
                         "Child Prior Placements Number": 0,
                         "Returning Child":	[True, False],
                         "Age Child Left Care":	[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
                         "Carer Age": [0, 0],
                         "Placement Type": ["Fostering - Long Term", "Fostering - Short Term", "Kinship",
                                            "Residential", "Special Guardianship", "Fostering - Emergency",
                                            "Fostering - Respite", "Adoption"],
                         "Missing Episodes": [0,1,2,3,4,5,6,7],
                         "Placement Sequence Number": [0],
                         "Placement Start Date": datetime.now(),
                         "Placement End Reason": "Reason why most recent placement ended",
                         "Placement End Time Period": "6 months ago",
                         "Number Of Carers": [0],
                         "Carer Gender Composition": [genders, genders],
                         "Placed With Siblings": [True, False],
                         "Emergency Placement": [True, False],
                         "Distance From Home": [0.0],
                         "Placement Planning Meeting": [True, False],
                         "Reason For Leaving Care":	"Why the child left care entirely",
                         "Sibling Group Size":	[0,1,2,3,4,5,6,7,8],
                         "Previous Care History": "Any additional context and comments about the child’s care history",
                         "Age Child Came Into Care": [0],
                         "Residential Home Type": "including Mainstream",
                         "Carer Ethnicity Or Religion":	[ethnicities, ethnicities],
                         "Carer Type": ["Long Term Foster", "Short Term Foster", "Kinship", "Residential",
                                        "Special Guardianship", "Emergency Foster", "Respite Foster", "Adoption"],
                         "EH involvement": [True, False],
                         "Siblings In EH": [True, False],
                         "YOT involvement": [True, False]
                         }
        self.__weighting = ["Eq", [0.05, 0, 0, 0, 0.36, 0, 0.1, 0.4, 0, 0.09],
                            [0, 0.002, 0.015, 0.002, 0.005, 0.013, 0.567, 0.004, 0.026, 0,
                             0.036, 0.005, 0, 0.255, 0.023, 0.002, 0.002, 0.012, 0, 0.005,
                             0.005, 0.002, 0.003, 0.036, 0.008, 0.015, 0.005, 0.036],
                            [0.5, 0.25, 0.12, 0.06, 0.03, 0.02, 0.01, 0.01],
                            [0.6, 0.4], "Eq", "Min 25, Max 75", "Eq", "Max 5",
                            [0, 0.5, 0.25, 0.12, 0.06, 0.03, 0.02, 0.01, 0.01],
                            "Eq", "Eq", "Eq", [0.5, 0.3, 0.15, 0.05],
                            [[0.05, 0, 0, 0, 0.36, 0, 0.1, 0.4, 0, 0.09], "Max 5"],
                            "Eq", "Eq", "Min 0.5, Max 1000.0", "Eq", "Eq",
                            [0.6, 0, 0.2, 0.1, 0.05, 0.03, 0.01, 0.01], "Eq", "Max 17", "Eq",
                            [[0, 0.002, 0.015, 0.002, 0.005, 0.013, 0.567, 0.004, 0.026, 0,
                             0.036, 0.005, 0, 0.255, 0.023, 0.002, 0.002, 0.012, 0, 0.005,
                             0.005, 0.002, 0.003, 0.036, 0.008, 0.015, 0.005, 0.036], "Max 5"],
                            "Eq", [0.3, 0.7], [0.25, 0.75], [0.1, 0.9]
                            ]

    def _resolve_weight_list(self, weighting):
        # If weighting is a nested list like [weights, "meta"], extract first numeric list
        if isinstance(weighting, list):
            # find the first sublist of numbers
            for item in weighting:
                if isinstance(item, list) and all(isinstance(x, (int, float)) for x in item):
                    return item
            # if weighting itself is numeric list use it
            if all(isinstance(x, (int, float)) for x in weighting):
                return weighting
        return None

    def _column_values(self, options, weighting, n):
        # constant (non-list) -> repeat
        if not isinstance(options, list):
            return [options] * n

        # Handle "Min X, Max Y" range
        if isinstance(weighting, str) and "Min" in weighting and "Max" in weighting:
            parts = weighting.replace("Min", "").replace("Max", "").split(",")
            min_val = float(parts[0].strip())
            max_val = float(parts[1].strip())
            return [random.uniform(min_val, max_val) for _ in range(n)]

        # Handle "Max X" range (0 to X inclusive for integers)
        if isinstance(weighting, str) and "Max" in weighting and "Min" not in weighting:
            max_val = int(weighting.replace("Max", "").strip())
            return [random.randint(0, max_val) for _ in range(n)]

        # try to get numeric weights
        weights = self._resolve_weight_list(weighting)
        # if we have valid numeric weights matching options length, use them
        if weights and len(weights) == len(options) and sum(weights) > 0:
            return random.choices(options, weights=weights, k=n)

        # If weighting explicitly "Eq" or no usable weights -> uniform random pick per-row
        if weighting == "Eq" or weights is None:
            return [random.choice(options) for _ in range(n)]

        # Fallback: uniform
        return [random.choice(options) for _ in range(n)]

    def create_dataset(self, num_rows=1000):
        headers = list(self.__fields.keys())

        # First pass: generate independent columns
        columns = {}
        for idx, key in enumerate(headers):
            options = self.__fields[key]
            weighting = None
            if idx < len(self.__weighting):
                weighting = self.__weighting[idx]
            columns[key] = self._column_values(options, weighting, num_rows)

        # Second pass: apply context-specific rules
        for i in range(num_rows):
            # Age Child Left Care > Child Age At Placement
            placement_age = columns["Child Age At Placement"][i]
            columns["Age Child Left Care"][i] = random.randint(placement_age + 1, 18)

            # Returning Child = False -> set N/A fields
            if not columns["Returning Child"][i]:
                columns["Placement End Reason"][i] = "N/A"
                columns["Placement End Time Period"][i] = "N/A"
                columns["Reason For Leaving Care"][i] = "N/A"
                columns["Previous Care History"][i] = "N/A"

            # Placement Sequence Number = Child Prior Placements Number + 1
            columns["Placement Sequence Number"][i] = columns["Child Prior Placements Number"][i] + 1

            # Match Placement Type with Carer Type
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

            # Number of Carers determines carer details
            num_carers = max(1, columns["Number Of Carers"][i])
            columns["Number Of Carers"][i] = num_carers

            # Round Distance From Home to 2 decimal places
            columns["Distance From Home"][i] = round(columns["Distance From Home"][i], 2)

            # Generate carer ages as integers
            columns["Carer Age"][i] = [random.randint(25, 75) for _ in range(num_carers)]

            # Generate carer genders
            genders = self.__fields["Child Gender"]
            columns["Carer Gender Composition"][i] = [random.choice(genders) for _ in range(num_carers)]

            # Generate carer ethnicities
            ethnicities = self.__fields["Child Ethnicity"]
            columns["Carer Ethnicity Or Religion"][i] = [random.choice(ethnicities) for _ in range(num_carers)]

            # Check sibling group size
            sibling_size = columns["Sibling Group Size"][i]
            if sibling_size > 1:
                columns["Placed With Siblings"][i] = True
            # Ensure sibling involvement in EH is not True if no siblings
            if sibling_size < 2:
                columns["Siblings In EH"][i] = False

        # Assemble row dicts
        data = []
        for i in range(num_rows):
            row = {key: columns[key][i] for key in headers}
            data.append(row)

        return data

    def write_dataset(self, data):
        file = open(self.__filePath, 'w')
        rowString = ""
        firstRow = True
        for row in list(data):
            rowString = ""
            if firstRow:
                firstRow = False
                for key in list(self.__fields.keys()):
                    rowString += key + ","
                file.write(rowString+"\n")
                rowString = ""
            if row != "":
                for value in list(row.values()):
                    rowString += str(value) + ","
                file.write(rowString+"\n")
        file.close()

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
    rows = dsb.create_dataset(1000)
    dsb.write_dataset(rows)
    print(f"Rows generated: {len(rows)}; First row below")
    # quick check: print all fields in the first row with the labels from DatasetBuilder fields
    for field in dsb.get_fields().keys():
        print(f"{field}: {rows[1][field]}")