select
    r.review_id,
    r.order_id,
    r.user_id::number as customer_id,
    try_to_number(r.restaurant_id) as restaurant_id,
    r.rating::number as rating,
    r.comment::string as comment,
    r.review_date::date as review_date,
    res.city as city
from {{ source('raw', 'reviews') }} r
left join {{ ref('stg_restaurants') }} res
    on try_to_number(r.restaurant_id) = res.restaurant_id
where r.comment is not null