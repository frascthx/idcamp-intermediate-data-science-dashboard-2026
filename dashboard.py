import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


@st.cache_data
def load_dataset(year: int | None = None):
    df = pd.read_csv('main_data.csv', parse_dates=['order_purchase_timestamp'])

    if year is not None:
        df = df[df['order_purchase_timestamp'].dt.year == year]

    return df


@st.cache_data
def get_unique_years():
    return [2018, 2017, 2016]
    # return sorted(main_df['order_purchase_timestamp'].dt.year.unique().tolist(), reverse=True)


@st.cache_data
def get_state_sales_revenue_df(year: int | None = None):
    result = (
        main_df[main_df['order_purchase_timestamp'].dt.year == year]
        .groupby('customer_city')
        .agg(
            total_sales=('order_id', 'count'),
            revenue=('price', 'sum')
        )
        .reset_index()
    )

    total_sales_mean = result['total_sales'].mean()

    revenue_mean = result['revenue'].mean()

    result = result[
        (result['total_sales'] > total_sales_mean) &
        (result['revenue'] < revenue_mean)
    ].sort_values(by='total_sales', ascending=False)

    result['avg_revenue_per_sales'] = (
        result['revenue']
        / result['total_sales']
    )

    return result


def render_high_sales_low_revenue_state(year: int | None = None):
    state_sales_revenue_result_df = get_state_sales_revenue_df(year=year)

    # set image size
    fig, ax = plt.subplots(figsize=(20, 12))

    # use sns scatterplot with size / buble chart
    sns.scatterplot(
        data=state_sales_revenue_result_df,
        x='total_sales',
        y='revenue',
        size='avg_revenue_per_sales',
        sizes=(500, 5000),
        alpha=0.7,
        edgecolor='black',
        legend=False,
        ax=ax
    )

    # focus on top 5 state with high sales low revenue
    volume_threshold = state_sales_revenue_result_df['total_sales'].quantile(
        0.75)

    focus_cities = (
        state_sales_revenue_result_df
        [state_sales_revenue_result_df['total_sales'] >= volume_threshold]
        .sort_values('avg_revenue_per_sales', ascending=True)
        .head(5)
    )

    # label only focus data
    for _, row in focus_cities.iterrows():
        ax.annotate(
            f"{row['customer_city']}\n${row['avg_revenue_per_sales']:.2f}",
            xy=(row['total_sales'], row['revenue']),
            xytext=(0, -(row['avg_revenue_per_sales']/2)-5),
            textcoords='offset points',
            ha='center',
            va='center',
            fontsize=12,
            weight='bold'
        )

    ax.set_xlabel('Total Sales Volume')
    ax.set_ylabel('Total Revenue')
    ax.set_title(f'City with High Sales but Low Revenue {year}')

    fig.tight_layout()
    st.pyplot(fig=fig)


@st.cache_data
def get_high_sales_low_revenue_product_df(year: int | None = None, top_n: int = 3):
    product_high_sales_low_revenue = (
        main_df[
            main_df['order_purchase_timestamp'].dt.year == year
        ]
        .groupby(['customer_city', 'product_category_name_english'])
        .agg(category_count=('product_category_name_english', 'count'))
        .sort_values(by='category_count', ascending=False)
        .reset_index()
    )

    state_sales_revenue_df = get_state_sales_revenue_df(year=year)

    top_5_cities = state_sales_revenue_df['customer_city'][:5].to_list()

    product_high_sales_low_revenue = product_high_sales_low_revenue[product_high_sales_low_revenue['customer_city'].isin(
        top_5_cities)]

    top3_per_city = (
        product_high_sales_low_revenue
        .sort_values(['customer_city', 'category_count'], ascending=[True, False])
        .groupby('customer_city')
        .head(top_n)
        .reset_index(drop=True)
    )

    return top3_per_city


def render_high_sales_low_revenue_product_category(year: int | None = None):
    top_n_product_filter = st.selectbox(
        label='Top N Product', options=[3, 4, 5])

    fig, ax = plt.subplots(figsize=(20, 16))

    sns.barplot(
        data=get_high_sales_low_revenue_product_df(
            year=year, top_n=top_n_product_filter),
        x='customer_city',
        y='category_count',
        hue='product_category_name_english',
        legend=True,
        ax=ax,
    )

    for container, category in zip(ax.containers, ax.get_legend_handles_labels()[1]):
        for bar in container:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height / 2,
                    category,
                    ha='center',
                    va='center',
                    fontsize=16,
                    color='white',
                    weight='bold',
                    rotation=90
                )

    ax.set_xlabel('City', fontsize=16)
    ax.set_ylabel('Sales Count', fontsize=16)
    ax.set_title(
        f'Top {top_n_product_filter} Product Categories per City\n with High Sales & Low Revenue {year}', fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.legend(title='Product Category')

    fig.tight_layout()
    st.pyplot(fig=fig)


def render_sales_by_name_length(year: int | None = None):
    sales_by_name_length = (
        main_df
        .groupby('product_id')
        .agg(
            name_length=('product_name_length', 'mean'),
            sales_volume=('order_id', 'count')
        )
        .reset_index()
    )

    fig, ax = plt.subplots()

    sns.scatterplot(data=sales_by_name_length,
                    x='name_length', y='sales_volume', ax=ax)

    ax.set_xlabel('Product Name Length')
    ax.set_ylabel('Sales Volume')
    ax.set_title(f'Sales Volume By Product Name Length {year}')

    st.pyplot(fig)


def render_sales_by_desc_length(year: int | None = None):
    sales_by_desc_length = (
        main_df
        .groupby('product_id')
        .agg(
            desc_length=('product_description_length', 'mean'),
            sales_volume=('order_id', 'count')
        )
        .reset_index()
    )

    fig, ax = plt.subplots()

    sns.scatterplot(data=sales_by_desc_length,
                    x='desc_length', y='sales_volume', ax=ax)

    ax.set_xlabel('Product Description Length')
    ax.set_ylabel('Sales Volume')
    ax.set_title(f'Sales Volume By Product Description Length {year}')

    st.pyplot(fig)


def render_sales_by_photo_count(year: int | None = None):
    sales_by_photo_count = (
        main_df
        .groupby('product_id')
        .agg(
            photo_count=('product_photos_qty', 'mean'),
            sales_volume=('order_id', 'count')
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    avg_sales_by_photo = (
        sales_by_photo_count.groupby('photo_count')[
            'sales_volume'].mean().reset_index()
    )

    # plt.figure(figsize=(10, 6))
    sns.barplot(data=avg_sales_by_photo,
                x='photo_count', y='sales_volume', ax=ax)
    ax.set_title(f'Average Sales Volume by Product Photo Count {year}')

    st.pyplot(fig)


def render_review_score_by_name_length(year: int | None = None):
    fig, ax = plt.subplots()

    sns.lineplot(data=main_df, x='product_name_length',
                 y='review_score', ax=ax)

    ax.set_xlabel('Product Name Length')

    ax.set_ylabel('Review Score')

    ax.set_title(f'Review Score by Product Name Length {year}')

    st.pyplot(fig)


def render_review_score_by_desc_length(year: int | None = None):
    fig, ax = plt.subplots()

    bins = [0, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000]

    labels = ["0-100", "101-200", "201-300", "301-400", "401-500",
              "501-1000", "1000-2000", "2001-3000", "3001-4000"]

    score_by_desc_bins = (
        main_df
        .assign(desc_bin=pd.cut(main_df['product_description_length'], bins=bins, labels=labels))
        .groupby('desc_bin')['review_score']
        .mean()
        .reset_index()
    )

    sns.lineplot(data=score_by_desc_bins, x='desc_bin',
                 y='review_score', marker='o', ax=ax)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_xlabel('Product Description Length')
    ax.set_ylabel('Review Score')
    ax.set_title(f'Review Score by Product Description Length {year}')

    st.pyplot(fig)
    
def render_review_score_by_photo_count(year: int | None = None):
    fig, ax = plt.subplots()

    sns.lineplot(data=main_df, x='product_photos_qty', y='review_score', ax=ax)
    ax.set_xlabel('Product Photo Count')
    ax.set_ylabel('Review Score')
    ax.set_title(f'Review Score by Product Photo Count {year}')

    st.pyplot(fig)
    

with st.sidebar:
    year_filter = st.selectbox(
        label='Year',
        options=get_unique_years()
    )
    
main_df = load_dataset(year_filter)

question_1_tab, question_2_tab = st.tabs(['Question 1', 'Question 2'])

with question_1_tab:
    st.header(
        body=f"Which states show high sales volume but contribute less to total revenue in {year_filter}, and what product categories dominate those states?")

    st.divider()

    st.subheader(body=f"State with High Sales but Low Revenue {year_filter}")

    render_high_sales_low_revenue_state(year=year_filter)

    st.divider()

    st.subheader(body=f"Dominating Product Categories {year_filter}")

    render_high_sales_low_revenue_product_category(year=year_filter)

with question_2_tab:
    st.header(
        body=f'How does product info (name length, desc length, photo count) influence sales volume and customer review scores for products sold in {year_filter}?')

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(body=f'Sales Volume By Product Name Length {year_filter}')

        render_sales_by_name_length(year=year_filter)

        st.divider()

        st.subheader(
            body=f'Sales Volume By Product Description Length {year_filter}')

        render_sales_by_desc_length(year=year_filter)

        st.divider()

        st.subheader(
            body=f'Average Sales Volume by Product Photo Count {year_filter}')

        render_sales_by_photo_count(year=year_filter)

    with col2:
        st.subheader(body=f'Review Score by Product Name Length {year_filter}')

        render_review_score_by_name_length(year=year_filter)

        st.divider()

        st.subheader(
            body=f'Review Score by Product Description Length {year_filter}')

        render_review_score_by_desc_length(year=year_filter)

        st.divider()
        
        st.subheader(
            body=f'Review Score by Product Photo Count {year_filter}')

        render_review_score_by_photo_count(year=year_filter)