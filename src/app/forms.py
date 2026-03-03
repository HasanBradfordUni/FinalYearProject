from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import StringField, PasswordField, SelectField, IntegerField, FloatField, TextAreaField, SubmitField, SelectMultipleField, BooleanField
from wtforms.validators import DataRequired, Email, Length, NumberRange, ValidationError, EqualTo, Optional

# ============== Authentication Forms ==============

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=50)])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Login')

class UserForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=50)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=8)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    role = SelectField('Role', choices=[('staff', 'Staff'), ('manager', 'Manager'), ('admin', 'Admin')], validators=[DataRequired()])
    submit = SubmitField('Create User')

class UserEditForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=50)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    role = SelectField('Role', choices=[('staff', 'Staff'), ('manager', 'Manager'), ('admin', 'Admin')], validators=[DataRequired()])
    is_active = BooleanField('Active')
    submit = SubmitField('Update User')

# ============== Placement Data Forms ==============

class PlacementUploadForm(FlaskForm):
    child_age = FloatField('Child Age at Placement', validators=[DataRequired(), NumberRange(min=0, max=18)])
    child_gender = SelectField('Child Gender',
                               choices=[('Male', 'Male'), ('Female', 'Female'), ('Non binary', 'Non-binary'),
                                       ('Trans Male', 'Trans Male'), ('Trans Female', 'Trans Female')],
                               validators=[DataRequired()])
    child_ethnicity = SelectField('Child Ethnicity',
                                  choices=[
                                      ('White - British', 'White - British'),
                                      ('White - Irish', 'White - Irish'),
                                      ('White - Other', 'White - Other'),
                                      ('White - Central European', 'White - Central European'),
                                      ('White - Eastern European', 'White - Eastern European'),
                                      ('Asian/British Asian - Indian', 'Asian/British Asian - Indian'),
                                      ('Asian/British Asian - Pakistani', 'Asian/British Asian - Pakistani'),
                                      ('Asian/British Asian - Bangladeshi', 'Asian/British Asian - Bangladeshi'),
                                      ('Asian/British Asian - Chinese', 'Asian/British Asian - Chinese'),
                                      ('Asian/British Asian - Other', 'Asian/British Asian - Other'),
                                      ('Black/Black British - African', 'Black/Black British - African'),
                                      ('Black/Black British - Caribbean', 'Black/Black British - Caribbean'),
                                      ('Black/Black British - Other', 'Black/Black British - Other'),
                                      ('Mixed - White/Black Caribbean', 'Mixed - White/Black Caribbean'),
                                      ('Mixed - White/Black African', 'Mixed - White/Black African'),
                                      ('Mixed - White/Asian', 'Mixed - White/Asian'),
                                      ('Mixed - Other', 'Mixed - Other'),
                                      ('Gypsy / Roma', 'Gypsy / Roma'),
                                      ('Traveller - Other', 'Traveller - Other'),
                                      ('Other Ethnic Group', 'Other Ethnic Group')
                                  ],
                                  validators=[DataRequired()])
    carer_age = FloatField('Carer Age', validators=[DataRequired(), NumberRange(min=18, max=100)])
    carer_gender = SelectField('Carer Gender Composition',
                              choices=[('Male', 'Male'), ('Female', 'Female'), ('Non binary', 'Non-binary'),
                                      ('Trans Male', 'Trans Male'), ('Trans Female', 'Trans Female')],
                              validators=[DataRequired()])
    carer_ethnicity = SelectField('Carer Ethnicity/Religion',
                                 choices=[
                                     ('White - British', 'White - British'),
                                     ('White - Irish', 'White - Irish'),
                                     ('White - Other', 'White - Other'),
                                     ('White - Central European', 'White - Central European'),
                                     ('White - Eastern European', 'White - Eastern European'),
                                     ('Asian/British Asian - Indian', 'Asian/British Asian - Indian'),
                                     ('Asian/British Asian - Pakistani', 'Asian/British Asian - Pakistani'),
                                     ('Asian/British Asian - Bangladeshi', 'Asian/British Asian - Bangladeshi'),
                                     ('Asian/British Asian - Chinese', 'Asian/British Asian - Chinese'),
                                     ('Asian/British Asian - Other', 'Asian/British Asian - Other'),
                                     ('Black/Black British - African', 'Black/Black British - African'),
                                     ('Black/Black British - Caribbean', 'Black/Black British - Caribbean'),
                                     ('Black/Black British - Other', 'Black/Black British - Other'),
                                     ('Mixed - White/Black Caribbean', 'Mixed - White/Black Caribbean'),
                                     ('Mixed - White/Black African', 'Mixed - White/Black African'),
                                     ('Mixed - White/Asian', 'Mixed - White/Asian'),
                                     ('Mixed - Other', 'Mixed - Other'),
                                     ('Gypsy / Roma', 'Gypsy / Roma'),
                                     ('Traveller - Other', 'Traveller - Other'),
                                     ('Other Ethnic Group', 'Other Ethnic Group')
                                 ],
                                 validators=[DataRequired()])
    placement_type = SelectField('Placement Type',
                                choices=[
                                    ('Fostering - Long Term', 'Fostering - Long Term'),
                                    ('Fostering - Short Term', 'Fostering - Short Term'),
                                    ('Fostering - Emergency', 'Fostering - Emergency'),
                                    ('Fostering - Respite', 'Fostering - Respite'),
                                    ('Kinship', 'Kinship'),
                                    ('Adoption', 'Adoption'),
                                    ('Residential', 'Residential'),
                                    ('Special Guardianship', 'Special Guardianship')
                                ],
                                validators=[DataRequired()])
    submit = SubmitField('Upload Placement')

class BulkUploadForm(FlaskForm):
    csv_file = FileField('CSV File', validators=[DataRequired(), FileAllowed(['csv'], 'CSV files only!')])
    submit = SubmitField('Upload CSV')

# ============== Prediction Forms ==============

class PredictionForm(FlaskForm):
    child_age = FloatField('Child Age at Placement', validators=[DataRequired(), NumberRange(min=0, max=18)])
    child_gender = SelectField('Child Gender',
                               choices=[('Male', 'Male'), ('Female', 'Female'), ('Non binary', 'Non-binary'),
                                       ('Trans Male', 'Trans Male'), ('Trans Female', 'Trans Female')],
                               validators=[DataRequired()])
    child_ethnicity = SelectField('Child Ethnicity',
                                  choices=[
                                      ('White - British', 'White - British'),
                                      ('White - Irish', 'White - Irish'),
                                      ('White - Other', 'White - Other'),
                                      ('White - Central European', 'White - Central European'),
                                      ('White - Eastern European', 'White - Eastern European'),
                                      ('Asian/British Asian - Indian', 'Asian/British Asian - Indian'),
                                      ('Asian/British Asian - Pakistani', 'Asian/British Asian - Pakistani'),
                                      ('Asian/British Asian - Bangladeshi', 'Asian/British Asian - Bangladeshi'),
                                      ('Asian/British Asian - Chinese', 'Asian/British Asian - Chinese'),
                                      ('Asian/British Asian - Other', 'Asian/British Asian - Other'),
                                      ('Black/Black British - African', 'Black/Black British - African'),
                                      ('Black/Black British - Caribbean', 'Black/Black British - Caribbean'),
                                      ('Black/Black British - Other', 'Black/Black British - Other'),
                                      ('Mixed - White/Black Caribbean', 'Mixed - White/Black Caribbean'),
                                      ('Mixed - White/Black African', 'Mixed - White/Black African'),
                                      ('Mixed - White/Asian', 'Mixed - White/Asian'),
                                      ('Mixed - Other', 'Mixed - Other'),
                                      ('Gypsy / Roma', 'Gypsy / Roma'),
                                      ('Traveller - Other', 'Traveller - Other'),
                                      ('Other Ethnic Group', 'Other Ethnic Group')
                                  ],
                                  validators=[DataRequired()])
    carer_age = FloatField('Carer Age', validators=[DataRequired(), NumberRange(min=18, max=100)])
    carer_gender = SelectField('Carer Gender Composition',
                              choices=[('Male', 'Male'), ('Female', 'Female'), ('Non binary', 'Non-binary'),
                                      ('Trans Male', 'Trans Male'), ('Trans Female', 'Trans Female')],
                              validators=[DataRequired()])
    carer_ethnicity = SelectField('Carer Ethnicity/Religion',
                                 choices=[
                                     ('White - British', 'White - British'),
                                     ('White - Irish', 'White - Irish'),
                                     ('White - Other', 'White - Other'),
                                     ('White - Central European', 'White - Central European'),
                                     ('White - Eastern European', 'White - Eastern European'),
                                     ('Asian/British Asian - Indian', 'Asian/British Asian - Indian'),
                                     ('Asian/British Asian - Pakistani', 'Asian/British Asian - Pakistani'),
                                     ('Asian/British Asian - Bangladeshi', 'Asian/British Asian - Bangladeshi'),
                                     ('Asian/British Asian - Chinese', 'Asian/British Asian - Chinese'),
                                     ('Asian/British Asian - Other', 'Asian/British Asian - Other'),
                                     ('Black/Black British - African', 'Black/Black British - African'),
                                     ('Black/Black British - Caribbean', 'Black/Black British - Caribbean'),
                                     ('Black/Black British - Other', 'Black/Black British - Other'),
                                     ('Mixed - White/Black Caribbean', 'Mixed - White/Black Caribbean'),
                                     ('Mixed - White/Black African', 'Mixed - White/Black African'),
                                     ('Mixed - White/Asian', 'Mixed - White/Asian'),
                                     ('Mixed - Other', 'Mixed - Other'),
                                     ('Gypsy / Roma', 'Gypsy / Roma'),
                                     ('Traveller - Other', 'Traveller - Other'),
                                     ('Other Ethnic Group', 'Other Ethnic Group')
                                 ],
                                 validators=[DataRequired()])
    submit = SubmitField('Generate Prediction')

class ComparisonForm(FlaskForm):
    child_age = FloatField('Child Age at Placement', validators=[DataRequired(), NumberRange(min=0, max=18)])
    child_gender = SelectField('Child Gender',
                               choices=[('Male', 'Male'), ('Female', 'Female'), ('Non binary', 'Non-binary'),
                                       ('Trans Male', 'Trans Male'), ('Trans Female', 'Trans Female')],
                               validators=[DataRequired()])
    child_ethnicity = SelectField('Child Ethnicity',
                                  choices=[
                                      ('White - British', 'White - British'),
                                      ('White - Irish', 'White - Irish'),
                                      ('White - Other', 'White - Other'),
                                      ('Asian/British Asian - Indian', 'Asian/British Asian - Indian'),
                                      ('Black/Black British - African', 'Black/Black British - African'),
                                      ('Other Ethnic Group', 'Other Ethnic Group')
                                  ],
                                  validators=[DataRequired()])
    carer_age = FloatField('Carer Age', validators=[DataRequired(), NumberRange(min=18, max=100)])
    carer_gender = SelectField('Carer Gender',
                              choices=[('Male', 'Male'), ('Female', 'Female'), ('Non binary', 'Non-binary')],
                              validators=[DataRequired()])
    carer_ethnicity = SelectField('Carer Ethnicity',
                                 choices=[
                                     ('White - British', 'White - British'),
                                     ('Asian/British Asian - Indian', 'Asian/British Asian - Indian'),
                                     ('Black/Black British - African', 'Black/Black British - African'),
                                     ('Other Ethnic Group', 'Other Ethnic Group')
                                 ],
                                 validators=[DataRequired()])
    placement_types = SelectMultipleField('Select Placement Types to Compare (2-4)',
                                         choices=[
                                             ('Fostering - Long Term', 'Fostering - Long Term'),
                                             ('Fostering - Short Term', 'Fostering - Short Term'),
                                             ('Fostering - Emergency', 'Fostering - Emergency'),
                                             ('Fostering - Respite', 'Fostering - Respite'),
                                             ('Kinship', 'Kinship'),
                                             ('Adoption', 'Adoption'),
                                             ('Residential', 'Residential'),
                                             ('Special Guardianship', 'Special Guardianship')
                                         ],
                                         validators=[DataRequired()])
    submit = SubmitField('Compare Placements')
