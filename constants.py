import dateutil

LABELS = ['Intern', 'Entry', 'Mid-Level', 'Senior', 'Manager', 'Director', 'Vice President', 'CXO']

CURRENT_DATE = dateutil.parser.parse('2020-01-01')
CURRENT_TIME_KEYWORDS = ['present', 'current', 'till date', 'now']

DEGREE_LIST = ['Bachelors', 'Masters', 'PhD', 'MBA', 'Associate', 'High School', 'Certification', 'Other']
MBA = ['mba', 'master of business', 'master in business', 'master business']
PHD = ['phd', 'doctor', 'dphil', 'dsc', 'dphil', 'md']
MASTERS = ['master', 'msc', 'mtech', 'meng', 'mca', 'mfa', 'mcom', 'mdes', 'mgs']
MASTERS_SHORT = ['ms', 'ma']
BACHELORS = ['bachelor', 'btech', 'bcom', 'bba', 'bfa', 'bdes', 'bgs', 'bca']
BACHELORS_SHORT = ['bs', 'ba', 'be']
ASSOCIATE = ['associate', 'aa']
HIGH_SCHOOL = ['high school', 'hs', 'hsc', 'hss', 'ssc', 'sslc', 'matriculation', 'matric', 'diploma']
CERTIFICATION = ['certific', 'ged']

CEO_KEYWORDS = ['ceo', 'chief executive officer', 'chief executive', 'chief', 'executive', 'officer', 'cto', 'chief technical officer', 'coo', 'chief operating officer', 'cfo', 'chief financial officer', 'cmo', 'chief marketing officer', 'cdo', 'chief data officer', 'cio', 'chief information officer', 'chairman']
VP_KEYWORDS = ['vp', 'vice president', 'vice', 'president']
DIRECTOR_KEYWORDS = ['director', 'dir']
INTERN_KEYWORDS = ['intern', 'internship', 'summer']

TITLE_KEYWORDS = [
    "intern",
    "internship",
    "trainee",
    "junior",
    "entry",
    "graduate",
    "associate",
    "assistant",
    "specialist",
    "coordinator",
    "mid",
    "intermediate",
    "senior",
    "sr.",
    "principal",
    "chief",
    "executive",
    "head",
    "director",
    "vp",
    "vice",
    "c-level",
    "cxo",
    "ceo",
    "cfo",
    "cto",
    "coo",
    "cmo",
    "president",
    "manager",
    "supervisor",
    "lead",
    "team",
    "global",
    "regional",
    "area",
    "divisional",
    "group",
    "board",
    "chairman",
    "partner",
    "founder",
    "owner",
    "officer",
]