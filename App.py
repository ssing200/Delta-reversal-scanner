@st.cache_data(ttl=CACHE_TTL)
def get_perpetual_products():
    result = api_get("/v2/products")

    columns = [
        "Coin",
        "ID",
        "Max Leverage",
        "Contract Value",
        "Tick Size",
    ]

    if not result:
        return pd.DataFrame(columns=columns)

    rows = []

    def first_value(item, keys):
        for key in keys:
            value = item.get(key)
            if value is not None and value != "":
                return value
        return None

    for item in result:
        if item.get("contract_type") != "perpetual_futures":
            continue

        if item.get("state") != "live":
            continue

        if item.get("trading_status") != "operational":
            continue

        symbol = item.get("symbol")
        if not symbol:
            continue

        # API versions may use different field names.
        leverage_raw = first_value(
            item,
            [
                "max_leverage",
                "maximum_leverage",
                "max_leverage_allowed",
                "leverage",
            ],
        )

        try:
            max_leverage = float(leverage_raw)
        except (TypeError, ValueError):
            max_leverage = np.nan

        contract_value = first_value(
            item,
            [
                "contract_value",
                "contract_value_usd",
                "contract_value_inr",
            ],
        )

        tick_size = first_value(
            item,
            [
                "tick_size",
                "price_band_tick_size",
            ],
        )

        rows.append(
            {
                "Coin": symbol,
                "ID": item.get("id"),
                "Max Leverage": max_leverage,
                "Contract Value": contract_value,
                "Tick Size": tick_size,
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns)

    df = (
        pd.DataFrame(rows)
        .drop_duplicates("Coin")
        .reset_index(drop=True)
    )

    df["Max Leverage"] = pd.to_numeric(
        df["Max Leverage"],
        errors="coerce",
    )

    return df
