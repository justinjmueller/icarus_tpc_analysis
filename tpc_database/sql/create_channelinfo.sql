CREATE TABLE IF NOT EXISTS channelinfo (
channel_id integer NOT NULL PRIMARY KEY,
tpc_number integer NOT NULL,
plane_number integer NOT NULL,
wire_number integer NOT NULL,
slot_id integer NOT NULL,
local_id integer NOT NULL,
group_id integer NOT NULL,
fragment_id integer NOT NULL,
flange_number integer NOT NULL,
flange_name text NOT NULL,
cable_number text NOT NULL,
channel_type text NOT NULL
);