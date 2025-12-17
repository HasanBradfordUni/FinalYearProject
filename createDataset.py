import random
from datetime import *

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
                       "Other Ethnic Group", "Dual Heritage - Black/White", "White - Eastern European"],
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
                            [0, 0.02, 0.03, 0.01, 0.02, 0.12, 0.3, 0.04, 0.06, 0,
                             0.03, 0.12, 0, 0.3, 0.1, 0.01, 0.01, 0.01, 0, 0.01,
                             0.01, 0.01, 0.01, 0.07, 0.03, 0.01, 0.01, 0.03],
                            [0.5, 0.25, 0.12, 0.06, 0.03, 0.02, 0.01, 0.01],
                            [0.6, 0.4], "Eq", "Min 25, Max 75", "Eq", "Max 5",
                            [0, 0.5, 0.25, 0.12, 0.06, 0.03, 0.02, 0.01, 0.01],
                            "Eq", "Eq", "Eq", [0.5, 0.3, 0.15, 0.05],
                            [[0.05, 0, 0, 0, 0.36, 0, 0.1, 0.4, 0, 0.09], "Max 5"],
                            "Eq", "Eq", "Min 0.5, Max 1000.0", "Eq", "Eq",
                            [0.6, 0, 0.2, 0.1, 0.05, 0.03, 0.01, 0.01], "Eq", "Eq", "Eq",
                            [[0, 0.02, 0.03, 0.01, 0.02, 0.12, 0.3, 0.04, 0.06, 0,
                             0.03, 0.12, 0, 0.3, 0.1, 0.01, 0.01, 0.01, 0, 0.01,
                             0.01, 0.01, 0.01, 0.07, 0.03, 0.01, 0.01, 0.03], "Max 5"],
                            "Eq", [0.3, 0.7], [0.25, 0.75], [0.1, 0.9]
                            ]

    def createDataset(self):
        data = []
        headers = []
        for index, key in enumerate(self.__fields):
            row = []
            headers.append(key)
            options = self.__fields[key]
            if self.__weighting[index] == "Eq" and isinstance(options, list):
                row.append(random.choice(options))
            elif isinstance(self.__weighting[index], list):
                for num in range(len(self.__weighting[index])):
                    weight = self.__weighting[index][num]
                    value = options[num]
                    print("The weight is", weight)
                    print("The value applied to is",value)

if __name__ == "__main__":
    dsb = DatasetBuilder()
    dsb.createDataset()



