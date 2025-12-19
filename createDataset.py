import random
from datetime import datetime

class DatasetBuilder:
    def __init__(self):
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
                         "Missing Episodes": 0,
                         "Placement Sequence Number": 0,
                         "Placement Start Date": datetime.now(),
                         "Placement End Reason": "Reason why most recent placement ended",
                         "Placement End Time Period": "6 months ago",
                         "Number Of Carers": 0,
                         "Carer Gender Composition": [genders, genders],
                         "Placed With Siblings": [True, False],
                         "Emergency Placement": [True, False],
                         "Distance From Home": 0.0,
                         "Placement Planning Meeting": [True, False],
                         "Reason For Leaving Care":	"Why the child left care entirely",
                         "Sibling Group Size":	0,
                         "Previous Care History": "Any additional context and comments about the child’s care history",
                         "Age Child Came Into Care": 0,
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
                            [0.6, 0, 0.2, 0.1, 0.05, 0.03, 0.01, 0.01], "Eq", "Eq", "Eq",
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

        # try to get numeric weights
        weights = self._resolve_weight_list(weighting)
        # if we have valid numeric weights matching options length, use them
        if weights and len(weights) == len(options) and sum(weights) > 0:
            # random.choices accepts raw weights; keep as provided (decimals summing ~1 are fine)
            return random.choices(options, weights=weights, k=n)

        # If weighting explicitly "Eq" or no usable weights -> uniform random pick per-row
        if weighting == "Eq" or weights is None:
            return [random.choice(options) for _ in range(n)]

        # Fallback: uniform
        return [random.choice(options) for _ in range(n)]

    def createDataset(self, num_rows=1000):
        headers = list(self.__fields.keys())
        # build columns for each field
        columns = {}
        for idx, key in enumerate(headers):
            options = self.__fields[key]
            weighting = None
            if idx < len(self.__weighting):
                weighting = self.__weighting[idx]
            columns[key] = self._column_values(options, weighting, num_rows)

        # assemble row dicts
        data = []
        for i in range(num_rows):
            row = {key: columns[key][i] for key in headers}
            data.append(row)

        return data

    def get_fields(self):
        return self.__fields

    def get_weights(self):
        return self.__weighting

if __name__ == "__main__":
    dsb = DatasetBuilder()
    rows = dsb.createDataset(1000)
    # quick check: count occurrences of first option for the first field
    first_field = list(dsb.get_fields().keys())[0]
    opt0 = dsb.get_fields()[first_field][0]
    count = sum(1 for r in rows if r[first_field] == opt0)
    print(f"Rows generated: {len(rows)}; occurrences of option 0 in '{first_field}': {count}")