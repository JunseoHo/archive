create database nexus_pcroom;
use nexus_pcroom;

create table user
(
    user_id varchar(30) primary key,
    user_pw varchar(30),
    user_name varchar(20),
    user_remaining_minute int unsigned
);

create table pc
(
    pc_id varchar(30) primary key,
    pc_logged_in_user varchar(30)
);


insert into user values('jho', '1234', 'JunseoHo', 0);
insert into user values('sejkim2', '1234', 'SejinKim', 0);
insert into user values('jaehyji', '1234', 'JaehyunJi', 0);
insert into user values('dongwook', '1234', 'Dongwookim', 0);

insert into pc values('PC001', null);
insert into pc values('PC002', null);
insert into pc values('PC003', null);
insert into pc values('PC004', null);
insert into pc values('PC005', null);
insert into pc values('PC006', null);
insert into pc values('PC007', null);
insert into pc values('PC008', null);