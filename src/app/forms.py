from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import StringField, PasswordField, SelectField, IntegerField, SubmitField, SelectMultipleField, BooleanField
from wtforms.validators import DataRequired, InputRequired, Email, Length, NumberRange, EqualTo, Optional
from .permissions import get_role_choices

GENDER_CHOICES = [
    ('Non binary', 'Non binary'),
    ('Male', 'Male'),
    ('Trans Female', 'Trans Female'),
    ('Female', 'Female'),
    ('Trans Male', 'Trans Male')
]

ETHNICITY_CHOICES = [
    ('Asian/British Asian - Chinese', 'Asian/British Asian - Chinese'),
    ('Asian/British Asian - Other', 'Asian/British Asian - Other'),
    ('Black/Black British - Other', 'Black/Black British - Other'),
    ('Gypsy / Roma', 'Gypsy / Roma'),
    ('Black/Black British - African', 'Black/Black British - African'),
    ('White - British', 'White - British'),
    ('White - Irish', 'White - Irish'),
    ('Asian/British Asian - Indian', 'Asian/British Asian - Indian'),
    ('White - Other', 'White - Other'),
    ('Black/Black British - Caribbean', 'Black/Black British - Caribbean'),
    ('Asian/British Asian - Pakistani', 'Asian/British Asian - Pakistani'),
    ('Asian/British Asian - Bangladeshi', 'Asian/British Asian - Bangladeshi'),
    ('Mixed - White/Black African', 'Mixed - White/Black African'),
    ('Traveller of Irish Heritage', 'Traveller of Irish Heritage'),
    ('Mixed - White/Asian', 'Mixed - White/Asian'),
    ('Mixed - Other', 'Mixed - Other'),
    ('Traveller - Other', 'Traveller - Other'),
    ('White - Central European', 'White - Central European'),
    ('Mixed - White/Black Caribbean', 'Mixed - White/Black Caribbean'),
    ('Dual Heritage - Black/White', 'Dual Heritage - Black/White'),
    ('White - Eastern European', 'White - Eastern European')
]

PLACEMENT_TYPE_CHOICES = [
    ('Kinship', 'Kinship'),
    ('External Fostering', 'External Fostering'),
    ('In-House Fostering', 'In-House Fostering')
]

BOOLEAN_CHOICES = [('True', 'Yes'), ('False', 'No')]
BOOLEAN_CHOICES_OPTIONAL = [('', 'Not provided')] + BOOLEAN_CHOICES
GENDER_CHOICES_OPTIONAL = [('', 'Not provided')] + GENDER_CHOICES
ETHNICITY_CHOICES_OPTIONAL = [('', 'Not provided')] + ETHNICITY_CHOICES

# ============== Authentication Forms ==============

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=50)])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Login')


class ForgotPasswordForm(FlaskForm):
    identifier = StringField('Username or Email', validators=[DataRequired(), Length(min=3, max=120)])
    submit = SubmitField('Generate Temporary Password')


class ChangePasswordForm(FlaskForm):
    current_password = PasswordField('Current Password', validators=[DataRequired()])
    new_password = PasswordField('New Password', validators=[DataRequired(), Length(min=8)])
    confirm_password = PasswordField('Confirm New Password', validators=[DataRequired(), EqualTo('new_password')])
    submit = SubmitField('Update Password')


class ResetPasswordForm(FlaskForm):
    new_password = PasswordField('New Password', validators=[DataRequired(), Length(min=8)])
    confirm_password = PasswordField('Confirm New Password', validators=[DataRequired(), EqualTo('new_password')])
    submit = SubmitField('Set New Password')

class UserForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=50)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=8)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    role = SelectField('Role', choices=get_role_choices(), validators=[DataRequired()])
    submit = SubmitField('Create User')

class UserEditForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=50)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    role = SelectField('Role', choices=get_role_choices(), validators=[DataRequired()])
    is_active = BooleanField('Active')
    submit = SubmitField('Update User')

# ============== Placement Data Forms ==============

class PlacementUploadForm(FlaskForm):
    child_age = IntegerField('Child Age at Placement', validators=[InputRequired(), NumberRange(min=0, max=17)], render_kw={'min': 0, 'max': 17, 'step': 1})
    child_gender = SelectField('Child Gender', choices=GENDER_CHOICES, validators=[DataRequired()])
    child_ethnicity = SelectField('Child Ethnicity', choices=ETHNICITY_CHOICES, validators=[DataRequired()])
    child_prior_placements = IntegerField('Child Prior Placements', validators=[InputRequired(), NumberRange(min=0, max=4)], render_kw={'min': 0, 'max': 4, 'step': 1})
    returning_child = SelectField('Returning Child', choices=BOOLEAN_CHOICES, validators=[DataRequired()])
    missing_episodes = IntegerField('Missing Episodes', validators=[InputRequired(), NumberRange(min=0, max=7)], render_kw={'min': 0, 'max': 7, 'step': 1})
    sibling_group_size = IntegerField('Sibling Group Size', validators=[InputRequired(), NumberRange(min=1, max=5)], render_kw={'min': 1, 'max': 5, 'step': 1, 'value': 1})
    placed_with_siblings = SelectField('Placed With Siblings', choices=BOOLEAN_CHOICES, validators=[DataRequired()])
    carer_age = IntegerField('Carer Age', validators=[InputRequired(), NumberRange(min=25, max=75)], render_kw={'min': 25, 'max': 75, 'step': 1})
    carer_gender = SelectField('Carer Gender', choices=GENDER_CHOICES, validators=[DataRequired()])
    carer_ethnicity = SelectField('Carer Ethnicity', choices=ETHNICITY_CHOICES, validators=[DataRequired()])
    eh_involvement = SelectField('EH involvement', choices=BOOLEAN_CHOICES, validators=[DataRequired()])
    yot_involvement = SelectField('YOT involvement', choices=BOOLEAN_CHOICES, validators=[DataRequired()])
    placement_type = SelectField('Placement Type', choices=PLACEMENT_TYPE_CHOICES, validators=[DataRequired()])
    submit = SubmitField('Upload Placement')

class BulkUploadForm(FlaskForm):
    csv_file = FileField('CSV File', validators=[DataRequired(), FileAllowed(['csv'], 'CSV files only!')])
    submit = SubmitField('Upload CSV')

# ============== Prediction Forms ==============

class PredictionForm(FlaskForm):
    child_age = IntegerField('Child Age at Placement', validators=[Optional(), NumberRange(min=0, max=17)], render_kw={'min': 0, 'max': 17, 'step': 1, 'placeholder': 'Leave blank to use average'})
    child_gender = SelectField('Child Gender', choices=GENDER_CHOICES_OPTIONAL, validators=[Optional()])
    child_ethnicity = SelectField('Child Ethnicity', choices=ETHNICITY_CHOICES_OPTIONAL, validators=[Optional()])
    child_prior_placements = IntegerField('Child Prior Placements', validators=[Optional(), NumberRange(min=0, max=4)], render_kw={'min': 0, 'max': 4, 'step': 1, 'placeholder': 'Leave blank to use average'})
    returning_child = SelectField('Returning Child', choices=BOOLEAN_CHOICES_OPTIONAL, validators=[Optional()])
    missing_episodes = IntegerField('Missing Episodes', validators=[Optional(), NumberRange(min=0, max=7)], render_kw={'min': 0, 'max': 7, 'step': 1, 'placeholder': 'Leave blank to use average'})
    sibling_group_size = IntegerField('Sibling Group Size', validators=[Optional(), NumberRange(min=1, max=5)], render_kw={'min': 1, 'max': 5, 'step': 1, 'placeholder': 'Leave blank to use average'})
    placed_with_siblings = SelectField('Placed With Siblings', choices=BOOLEAN_CHOICES_OPTIONAL, validators=[Optional()])
    carer_age = IntegerField('Carer Age', validators=[Optional(), NumberRange(min=25, max=75)], render_kw={'min': 25, 'max': 75, 'step': 1, 'placeholder': 'Leave blank to use average'})
    carer_gender = SelectField('Carer Gender', choices=GENDER_CHOICES_OPTIONAL, validators=[Optional()])
    carer_ethnicity = SelectField('Carer Ethnicity', choices=ETHNICITY_CHOICES_OPTIONAL, validators=[Optional()])
    eh_involvement = SelectField('EH involvement', choices=BOOLEAN_CHOICES_OPTIONAL, validators=[Optional()])
    yot_involvement = SelectField('YOT involvement', choices=BOOLEAN_CHOICES_OPTIONAL, validators=[Optional()])
    submit = SubmitField('Generate Prediction')

class ComparisonForm(FlaskForm):
    child_age = IntegerField('Child Age at Placement', validators=[Optional(), NumberRange(min=0, max=17)], render_kw={'min': 0, 'max': 17, 'step': 1, 'placeholder': 'Leave blank to use average'})
    child_gender = SelectField('Child Gender', choices=GENDER_CHOICES_OPTIONAL, validators=[Optional()])
    child_ethnicity = SelectField('Child Ethnicity', choices=ETHNICITY_CHOICES_OPTIONAL, validators=[Optional()])
    child_prior_placements = IntegerField('Child Prior Placements', validators=[Optional(), NumberRange(min=0, max=4)], render_kw={'min': 0, 'max': 4, 'step': 1, 'placeholder': 'Leave blank to use average'})
    returning_child = SelectField('Returning Child', choices=BOOLEAN_CHOICES_OPTIONAL, validators=[Optional()])
    missing_episodes = IntegerField('Missing Episodes', validators=[Optional(), NumberRange(min=0, max=7)], render_kw={'min': 0, 'max': 7, 'step': 1, 'placeholder': 'Leave blank to use average'})
    sibling_group_size = IntegerField('Sibling Group Size', validators=[Optional(), NumberRange(min=1, max=5)], render_kw={'min': 1, 'max': 5, 'step': 1, 'placeholder': 'Leave blank to use average'})
    placed_with_siblings = SelectField('Placed With Siblings', choices=BOOLEAN_CHOICES_OPTIONAL, validators=[Optional()])
    carer_age = IntegerField('Carer Age', validators=[Optional(), NumberRange(min=25, max=75)], render_kw={'min': 25, 'max': 75, 'step': 1, 'placeholder': 'Leave blank to use average'})
    carer_gender = SelectField('Carer Gender', choices=GENDER_CHOICES_OPTIONAL, validators=[Optional()])
    carer_ethnicity = SelectField('Carer Ethnicity', choices=ETHNICITY_CHOICES_OPTIONAL, validators=[Optional()])
    eh_involvement = SelectField('EH involvement', choices=BOOLEAN_CHOICES_OPTIONAL, validators=[Optional()])
    yot_involvement = SelectField('YOT involvement', choices=BOOLEAN_CHOICES_OPTIONAL, validators=[Optional()])
    placement_types = SelectMultipleField('Select Placement Types to Compare (2-4)',
                                         choices=PLACEMENT_TYPE_CHOICES,
                                         validators=[DataRequired()])
    submit = SubmitField('Compare Placements')
