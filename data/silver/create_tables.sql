create schema if not exists silver;

--1. when can you trade, when the exchange open or close
create table if not exists silver.exchange_sessions(

    session_date    DATE NOT NULL,
    market_open_at  TIMESTAMPTZ NOT NULL,
    market_close_at TIMESTAMPTZ NOT NULL,

    check(market_close_at>market_open_at)
);

--2.the map between company and the stocks
create table if not exists silver.entity_security_links(
    issuer_id       varchar not null,
    security_id     varchar not null,

    valid_from      DATE not null,
    valid_to        DATE,

    primary key(
        issuer_id,
        security_id,
        valid_from,
    ),

    check(valid_to > valid_from)

);

--3. the map between stock and tickers

create table if not exists silver.listing_history(
    listing_id      varchar not null,
    security_id     varchar not null,
    
    ticker          varchar not null,

    valid_from      DATE not null,
    valid_to        DATE,

    primary key(
        listing_id,
        valid_from
    ),

    check(valid_to > valid_from)
);


--4.daily bars
create table if not exists silver.daily_bars(
    listing_id          VARCHAR NOT NULL,

    session_date        DATE NOT NULL,
    open_price          DOUBLE,
    low_price           DOUBLE,
    close_price         DOUBLE,

    volume_raw          BIGINT,
    primary key(
        listing_id,
        session_date
    ),

    check(
        open_price>0
        and close_price>0
        and volume_raw>=0

    )
)
;

--5.event of the company
create table if not exists silver.entity_security_links(
    link_id             VARCHAR PRIMARY KEY,
    security_id         VARCHAR NOT NULL,
    currency            VARCHAR NOT NULL DEFAULT 'USD',

)
;
create table if not exists silver.events(
    event_id            VARCHAR PRIMARY KEY,
    issuer_id            VARCHAR NOT NULL,

    event_type          VARCHAR NOT NULL,

    available_at        TIMESTAMPTZ NOT NULL,
);