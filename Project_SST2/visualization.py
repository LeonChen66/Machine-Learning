import pandas as pd
import sys
import io
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from pylab import *
from scipy.stats import norm
from scipy.stats import hmean
import seaborn as sns
from bokeh.plotting import figure
from bokeh.io import output_notebook, show
from bokeh.models import LinearColorMapper
from bokeh.models import HoverTool
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')


def read_SST2(filename):
    df = pd.read_table(filename)
    return df


def normcdf(x):
    return norm.cdf(x, x.mean(), x.std())

def Vectorizer(df,stop_word=False):
    if stop_word==True:
        cvec = CountVectorizer(stop_words='english')
    else:
        cvec = CountVectorizer()

    cvec.fit(df['sentence'])
    neg_doc_matrix = cvec.transform(df[df.label == 0].sentence)
    pos_doc_matrix = cvec.transform(df[df.label == 1].sentence)
    neg_tf = np.sum(neg_doc_matrix, axis=0)
    pos_tf = np.sum(pos_doc_matrix, axis=0)
    neg = np.squeeze(np.asarray(neg_tf))
    pos = np.squeeze(np.asarray(pos_tf))
    term_freq_df = pd.DataFrame(
        [neg, pos], columns=cvec.get_feature_names()).transpose()
    term_freq_df.columns = ['negative', 'positive']
    term_freq_df['total'] = term_freq_df['negative'] + term_freq_df['positive']
    term_freq_df.sort_values(by='total', ascending=False)
    term_freq_df['pos_rate'] = term_freq_df['positive'] * \
        1./term_freq_df['total']
    term_freq_df['pos_freq_pct'] = term_freq_df['positive'] * \
        1./term_freq_df['positive'].sum()
    term_freq_df['pos_hmean'] = term_freq_df.apply(lambda x: (hmean([x['pos_rate'], x['pos_freq_pct']])
                                if x['pos_rate'] > 0 and x['pos_freq_pct'] > 0
                                else 0), axis=1)

    term_freq_df['pos_rate_normcdf'] = normcdf(term_freq_df['pos_rate'])
    term_freq_df['pos_freq_pct_normcdf'] = normcdf(
                                             term_freq_df['pos_freq_pct'])
    term_freq_df['pos_normcdf_hmean'] = hmean([term_freq_df['pos_rate_normcdf'], 
                                            term_freq_df['pos_freq_pct_normcdf']])
    term_freq_df['neg_rate'] = term_freq_df['negative'] * \
        1./term_freq_df['total']
    term_freq_df['neg_freq_pct'] = term_freq_df['negative'] * \
        1./term_freq_df['negative'].sum()
    term_freq_df['neg_hmean'] = term_freq_df.apply(lambda x: (hmean([x['neg_rate'], x['neg_freq_pct']])
                                                                    if x['neg_rate'] > 0 and x['neg_freq_pct'] > 0 
                                                                    else 0), axis=1)                                                        
    term_freq_df['neg_rate_normcdf'] = normcdf(term_freq_df['neg_rate'])
    term_freq_df['neg_freq_pct_normcdf'] = normcdf(
        term_freq_df['neg_freq_pct'])
    term_freq_df['neg_normcdf_hmean'] = hmean(
        [term_freq_df['neg_rate_normcdf'], term_freq_df['neg_freq_pct_normcdf']])
    return term_freq_df


def top_500_tokens(term_freq_df):
    y_pos = np.arange(500)
    plt.figure(figsize=(10, 8))
    s = 1
    expected_zipf = [term_freq_df.sort_values(by='total', ascending=False)[
        'total'][0]/(i+1)**s for i in y_pos]

    plt.bar(y_pos, term_freq_df.sort_values(by='total', ascending=False)
            ['total'][:500], align='center', alpha=0.5)
    plt.plot(y_pos, expected_zipf, color='r',
            linestyle='--', linewidth=2, alpha=0.5)
    plt.ylabel('Frequency')
    plt.title('Top 500 tokens in Moive Review')
    plt.show()

def zipf_plot(term_freq_df):
    counts = term_freq_df.total
    tokens = term_freq_df.index
    ranks = arange(1, len(counts)+1)
    indices = argsort(-counts)
    frequencies = counts[indices]
    plt.figure(figsize=(8, 6))
    plt.ylim(1, 10**6)
    plt.xlim(1, 10**6)
    loglog(ranks, frequencies, marker=".")
    plt.plot([1, frequencies[0]], [frequencies[0], 1], color='r')
    title("Zipf plot for tweets tokens")
    xlabel("Frequency rank of token")
    ylabel("Absolute frequency of token")
    grid(True)
    for n in list(logspace(-0.5, log10(len(counts)-2), 25).astype(int)):
        dummy = text(ranks[n], frequencies[n], " " + tokens[indices[n]],
                    verticalalignment="bottom",
                    horizontalalignment="left")
    plt.show()


def top_50_tokens(term_freq_df,mode='positive'):
    if mode == 'positive':
        y_pos = np.arange(50)
        plt.figure(figsize=(12, 10))
        plt.bar(y_pos, term_freq_df.sort_values(by='positive', ascending=False)[
                'positive'][:50], align='center', alpha=0.5)
        plt.xticks(y_pos, term_freq_df.sort_values(by='positive', ascending=False)[
                'positive'][:50].index, rotation='vertical')
        plt.ylabel('Frequency')
        plt.xlabel('Top 50 positive tokens')
        plt.title('Top 50 positive tokens in  Movie Review')

    elif mode== 'negative':
        y_pos = np.arange(50)
        plt.figure(figsize=(12, 10))
        plt.bar(y_pos, term_freq_df.sort_values(by='negative', ascending=False)[
                'negative'][:50], align='center', alpha=0.5)
        plt.xticks(y_pos, term_freq_df.sort_values(by='negative', ascending=False)[
                'negative'][:50].index, rotation='vertical')
        plt.ylabel('Frequency')
        plt.xlabel('Top 50 negative tokens')
        plt.title('Top 50 negative tokens in Movie Review')
        plt.show()


def nega_vs_pos(term_freq_df):
    plt.figure(figsize=(8, 6))
    ax = sns.regplot(x="negative", y="positive", fit_reg=False,
                    scatter_kws={'alpha': 0.5}, data=term_freq_df)
    plt.ylabel('Positive Frequency')
    plt.xlabel('Negative Frequency')
    plt.title('Negative Frequency vs Positive Frequency')
    plt.show()


def correlation(term_freq_df,mode='seaborn'):
    if mode=='seaborn':
        plt.figure(figsize=(8, 6))
        ax = sns.regplot(x="neg_normcdf_hmean", y="pos_normcdf_hmean",
                        fit_reg=False, scatter_kws={'alpha': 0.5}, data=term_freq_df)
        plt.ylabel('Positive Rate and Frequency CDF Harmonic Mean')
        plt.xlabel('Negative Rate and Frequency CDF Harmonic Mean')
        plt.title('neg_normcdf_hmean vs pos_normcdf_hmean')
        plt.show()
    elif mode=='D3js':
        output_notebook()
        color_mapper = LinearColorMapper(palette='Inferno256', low=min(
            term_freq_df.pos_normcdf_hmean), high=max(term_freq_df.pos_normcdf_hmean))
        p = figure(title="Pos CDF vs Neg CDF", x_axis_label='neg_normcdf_hmean',
                   y_axis_label='pos_normcdf_hmean')
        p.circle('neg_normcdf_hmean', 'pos_normcdf_hmean', size=5, alpha=0.3,
                source=term_freq_df, color={'field': 'pos_normcdf_hmean', 'transform': color_mapper})
        hover = HoverTool(tooltips=[('token', '@index')])
        p.add_tools(hover)
        show(p)
    else:
        print('Not an option')
    
if __name__ == "__main__":
    df = read_SST2('SST2/train.tsv')
    term_freq_df = Vectorizer(df,True)
    # top_500_tokens(term_freq_df)
    # top_50_tokens(term_freq_df, mode='positive')
    # top_50_tokens(term_freq_df,mode='negative')
    nega_vs_pos(term_freq_df)
    # zipf_plot(term_freq_df)
    # correlation(term_freq_df)




