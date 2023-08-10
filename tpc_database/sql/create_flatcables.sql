CREATE TABLE IF NOT EXISTS flatcables (
cable_number text NOT NULL PRIMARY KEY,
plane_type text NOT NULL,
crimp_style_at_ft text NOT NULL,
crimp_style_at_wire text NOT NULL,
geometric_length real NOT NULL,
cable_length_code integer NOT NULL,
real_length real NOT NULL,
capacitance real NOT NULL
);