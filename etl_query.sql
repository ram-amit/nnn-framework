-- =============================================================================
-- NNN Framework: Weekly Marketing Data ETL Query
-- Dialect: BigQuery (Snowflake alternatives in inline comments)
--
-- Joins ad platform performance tables + CRM conversions, aggregates to
-- weekly grain, and outputs the exact schema data_prep.py expects:
--   Date, Geography, Channel, Spend, Enterprise_Trials, Closed_Won, Campaign_Metadata
--
-- Assumptions:
--   - Ad platform tables land via Fivetran/Airbyte with standard schemas
--   - CRM conversions are attributed to campaigns via campaign_id
--   - "Week" = ISO week starting Monday
-- =============================================================================

WITH

-- ─────────────────────────────────────────────
-- 1. Geography mapping: country → region
-- ─────────────────────────────────────────────
geo_map AS (
    SELECT * FROM UNNEST([
        STRUCT('US' AS country_code, 'NA' AS geography),
        STRUCT('CA' AS country_code, 'NA' AS geography),
        STRUCT('MX' AS country_code, 'NA' AS geography),
        STRUCT('GB' AS country_code, 'EMEA' AS geography),
        STRUCT('DE' AS country_code, 'EMEA' AS geography),
        STRUCT('FR' AS country_code, 'EMEA' AS geography),
        STRUCT('NL' AS country_code, 'EMEA' AS geography),
        STRUCT('IL' AS country_code, 'EMEA' AS geography),
        STRUCT('SE' AS country_code, 'EMEA' AS geography),
        STRUCT('AU' AS country_code, 'APAC' AS geography),
        STRUCT('JP' AS country_code, 'APAC' AS geography),
        STRUCT('SG' AS country_code, 'APAC' AS geography),
        STRUCT('IN' AS country_code, 'APAC' AS geography)
    ])
    -- Snowflake: replace with a VALUES clause or a reference table
    -- SELECT country_code, geography FROM analytics.dim_geography
),

-- ─────────────────────────────────────────────
-- 2. LinkedIn Ads — daily → weekly
-- ─────────────────────────────────────────────
linkedin_raw AS (
    SELECT
        -- Snowflake: DATE_TRUNC('WEEK', date)
        DATE_TRUNC(date, ISOWEEK)                                   AS week_start,
        g.geography,
        'LinkedIn Ads'                                               AS channel,
        SUM(spend)                                                   AS spend,
        SUM(conversions)                                             AS enterprise_trials,
        -- Metadata: one row per campaign-week-geo, aggregate with ANY_VALUE
        -- since campaign-level text is constant within a campaign
        CONCAT(
            '[Campaign: ', ANY_VALUE(campaign_name), '] ',
            '[Copy: ',     ANY_VALUE(creative_text), '] ',
            '[Audience: ', ANY_VALUE(audience_targeting), '] ',
            '[Objective: ', ANY_VALUE(objective), ']'
        )                                                            AS campaign_metadata
    FROM `marketing_dwh.linkedin_ads_performance` li
    LEFT JOIN geo_map g ON li.country_code = g.country_code
    WHERE date >= '2024-01-01'
    GROUP BY 1, 2, 3
),

-- ─────────────────────────────────────────────
-- 3. Google Ads — split into Brand Search, Non-Brand Search,
--    Display, and YouTube based on campaign_type + naming convention
-- ─────────────────────────────────────────────
google_raw AS (
    SELECT
        DATE_TRUNC(date, ISOWEEK)                                   AS week_start,
        g.geography,
        CASE
            WHEN campaign_type = 'SEARCH'
                 AND LOWER(campaign_name) LIKE '%brand%'
                THEN 'Google Brand'
            WHEN campaign_type = 'SEARCH'
                THEN 'Google Non-Brand'
            WHEN campaign_type = 'DISPLAY'
                THEN 'Google Display'
            WHEN campaign_type = 'VIDEO'
                THEN 'YouTube'
            ELSE CONCAT('Google ', INITCAP(campaign_type))
        END                                                          AS channel,
        SUM(cost_micros / 1e6)                                       AS spend,
        SUM(conversions)                                             AS enterprise_trials,
        CONCAT(
            '[Campaign: ', ANY_VALUE(campaign_name), '] ',
            '[Copy: ',     ANY_VALUE(COALESCE(headline, '')),
                      ' | ', ANY_VALUE(COALESCE(description, '')), '] ',
            '[Audience: ', ANY_VALUE(COALESCE(audience_segment, 'Broad')), '] ',
            '[Objective: ', ANY_VALUE(COALESCE(bidding_strategy, 'Maximize Conversions')), ']'
        )                                                            AS campaign_metadata
    FROM `marketing_dwh.google_ads_performance` ga
    LEFT JOIN geo_map g ON ga.country_code = g.country_code
    WHERE date >= '2024-01-01'
    GROUP BY 1, 2, 3
),

-- ─────────────────────────────────────────────
-- 4. Content Syndication (e.g., TechTarget, Bombora)
-- ─────────────────────────────────────────────
content_synd_raw AS (
    SELECT
        DATE_TRUNC(delivery_date, ISOWEEK)                           AS week_start,
        g.geography,
        'Content Syndication'                                        AS channel,
        SUM(cost)                                                    AS spend,
        SUM(leads_delivered)                                         AS enterprise_trials,
        CONCAT(
            '[Campaign: ', ANY_VALUE(program_name), '] ',
            '[Copy: ',     ANY_VALUE(asset_title), '] ',
            '[Audience: ', ANY_VALUE(targeting_criteria), '] ',
            '[Objective: Cost-per-lead]'
        )                                                            AS campaign_metadata
    FROM `marketing_dwh.content_syndication_performance` cs
    LEFT JOIN geo_map g ON cs.country_code = g.country_code
    WHERE delivery_date >= '2024-01-01'
    GROUP BY 1, 2, 3
),

-- ─────────────────────────────────────────────
-- 5. CRM closed-won deals, attributed to channel via campaign source
-- ─────────────────────────────────────────────
crm_closed_won AS (
    SELECT
        DATE_TRUNC(o.close_date, ISOWEEK)                           AS week_start,
        g.geography,
        CASE
            WHEN LOWER(c.campaign_source) LIKE '%linkedin%'     THEN 'LinkedIn Ads'
            WHEN LOWER(c.campaign_source) LIKE '%brand%'
                 AND LOWER(c.campaign_source) LIKE '%google%'   THEN 'Google Brand'
            WHEN LOWER(c.campaign_source) LIKE '%google%search%'
                 OR LOWER(c.campaign_source) LIKE '%sem%'       THEN 'Google Non-Brand'
            WHEN LOWER(c.campaign_source) LIKE '%display%'
                 OR LOWER(c.campaign_source) LIKE '%gdn%'       THEN 'Google Display'
            WHEN LOWER(c.campaign_source) LIKE '%youtube%'
                 OR LOWER(c.campaign_source) LIKE '%video%'     THEN 'YouTube'
            WHEN LOWER(c.campaign_source) LIKE '%content%'
                 OR LOWER(c.campaign_source) LIKE '%syndication%' THEN 'Content Syndication'
            ELSE 'Other'
        END                                                          AS channel,
        COUNT(DISTINCT o.opportunity_id)                              AS closed_won
    FROM `crm_dwh.salesforce_opportunities` o
    LEFT JOIN `crm_dwh.salesforce_campaign_members` cm
        ON o.contact_id = cm.contact_id
    LEFT JOIN `crm_dwh.salesforce_campaigns` c
        ON cm.campaign_id = c.campaign_id
    LEFT JOIN geo_map g ON o.account_country_code = g.country_code
    WHERE o.stage_name = 'Closed Won'
      AND o.close_date >= '2024-01-01'
    GROUP BY 1, 2, 3
),

-- ─────────────────────────────────────────────
-- 6. Union all channels into a single spine
-- ─────────────────────────────────────────────
all_channels AS (
    SELECT week_start, geography, channel, spend, enterprise_trials, campaign_metadata
    FROM linkedin_raw
    UNION ALL
    SELECT week_start, geography, channel, spend, enterprise_trials, campaign_metadata
    FROM google_raw
    UNION ALL
    SELECT week_start, geography, channel, spend, enterprise_trials, campaign_metadata
    FROM content_synd_raw
),

-- ─────────────────────────────────────────────
-- 7. Join with CRM closed-won counts
-- ─────────────────────────────────────────────
final AS (
    SELECT
        FORMAT_DATE('%Y-%m-%d', ac.week_start)                       AS Date,
        -- Snowflake: TO_CHAR(ac.week_start, 'YYYY-MM-DD')
        ac.geography                                                 AS Geography,
        ac.channel                                                   AS Channel,
        ROUND(ac.spend, 2)                                           AS Spend,
        ac.enterprise_trials                                         AS Enterprise_Trials,
        COALESCE(cw.closed_won, 0)                                   AS Closed_Won,
        ac.campaign_metadata                                         AS Campaign_Metadata
    FROM all_channels ac
    LEFT JOIN crm_closed_won cw
        ON ac.week_start = cw.week_start
       AND ac.geography  = cw.geography
       AND ac.channel    = cw.channel
    WHERE ac.geography IS NOT NULL
)

SELECT * FROM final
ORDER BY Date, Geography, Channel
