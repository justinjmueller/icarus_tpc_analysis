CREATE TABLE IF NOT EXISTS logicalwires(
channel_id integer NOT NULL,
wire_number integer NOT NULL,
c integer NOT NULL,
t integer NOT NULL,
p integer NOT NULL,
PRIMARY KEY (channel_id, t)
);