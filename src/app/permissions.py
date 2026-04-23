"""Centralized role and permission mapping for route and template access checks."""

ROLE_LABELS = {
    "it_app_support": "IT & App Support",
    "data_performance_team": "Data & Performance Team",
    "placement_officer": "Placement Officer",
    "social_work_team": "Social Work Team",
    "service_manager": "Service Manager & Team Manager",
    "residential_placement_lead": "Residential Placement Lead",
}

# Backward-compatible aliases for legacy records.
ROLE_ALIASES = {
    "admin": "it_app_support",
    "manager": "service_manager",
    "staff": "placement_officer",
    "data_analytics_team": "data_performance_team",
    "data_and_performance_team": "data_performance_team",
    "data_analyst": "data_performance_team",
    "social_worker": "social_work_team",
    "social_workers": "social_work_team",
    "placement_officers": "placement_officer",
    "residential_lead": "residential_placement_lead",
}

ADMIN_ROLES = {"it_app_support", "data_performance_team"}

PERMISSIONS = {
    "access_staff_dashboard": {
        "placement_officer",
        "social_work_team",
        "service_manager",
        "residential_placement_lead",
        "it_app_support",
        "data_performance_team",
    },
    "access_manager_dashboard": {
        "service_manager",
        "residential_placement_lead",
        "it_app_support",
        "data_performance_team",
    },
    "access_admin_dashboard": ADMIN_ROLES,
    "upload_placements": {
        "placement_officer",
        "residential_placement_lead",
        "it_app_support",
        "data_performance_team",
    },
    "predict": {
        "placement_officer",
        "social_work_team",
        "service_manager",
        "residential_placement_lead",
        "it_app_support",
        "data_performance_team",
    },
    "compare": {
        "placement_officer",
        "service_manager",
        "residential_placement_lead",
        "it_app_support",
        "data_performance_team",
    },
    "breakdown_read": {
        "placement_officer",
        "social_work_team",
        "service_manager",
        "residential_placement_lead",
        "it_app_support",
        "data_performance_team",
    },
    "breakdown_full": {
        "service_manager",
        "residential_placement_lead",
        "it_app_support",
        "data_performance_team",
    },
    "breakdown_export": {
        "service_manager",
        "residential_placement_lead",
        "it_app_support",
        "data_performance_team",
    },
    "stability_trends": {
        "service_manager",
        "residential_placement_lead",
        "it_app_support",
        "data_performance_team",
    },
    "view_placement_record": {
        "placement_officer",
        "social_work_team",
        "service_manager",
        "residential_placement_lead",
        "it_app_support",
        "data_performance_team",
    },
    "view_all_placements": {
        "service_manager",
        "residential_placement_lead",
        "it_app_support",
        "data_performance_team",
    },
    "export_ai_outputs": {
        "placement_officer",
        "service_manager",
        "residential_placement_lead",
        "it_app_support",
        "data_performance_team",
    },
    "admin_manage_system": ADMIN_ROLES,
    "view_audit_logs": {
        "service_manager",
        "it_app_support",
        "data_performance_team",
    },
}


def normalize_role(role):
    if not role:
        return role
    normalized = str(role).strip().lower().replace('&', 'and').replace('-', ' ').replace('/', ' ')
    normalized = '_'.join(normalized.split())
    normalized = ROLE_ALIASES.get(normalized, normalized)
    if normalized in ROLE_LABELS:
        return normalized

    label_lookup = {
        '_'.join(value.lower().replace('&', 'and').replace('-', ' ').replace('/', ' ').split()): key
        for key, value in ROLE_LABELS.items()
    }
    return label_lookup.get(normalized, normalized)


def is_admin_role(role):
    return normalize_role(role) in ADMIN_ROLES


def has_permission(role, permission):
    normalized_role = normalize_role(role)
    return normalized_role in PERMISSIONS.get(permission, set())


def get_role_choices():
    return [(key, label) for key, label in ROLE_LABELS.items()]
