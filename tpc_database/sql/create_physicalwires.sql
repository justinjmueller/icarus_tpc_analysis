CREATE TABLE IF NOT EXISTS physicalwires(
channel_id integer NOT NULL PRIMARY KEY,
x float NOT NULL,
y0 float NOT NULL,
z0 float NOT NULL,
y1 float NOT NULL,
z1 float NOT NULL,
length float NOT NULL,
capacitance NOT NULL
);