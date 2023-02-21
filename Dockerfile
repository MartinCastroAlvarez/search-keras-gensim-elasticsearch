FROM docker.elastic.co/elasticsearch/elasticsearch:7.9.2

# Installing the Keras plugin.
#
# Reference:
# - http://es-learn-to-rank.labs.o19s.com/
RUN /usr/share/elasticsearch/bin/elasticsearch-plugin install analysis-icu
RUN /usr/share/elasticsearch/bin/elasticsearch-plugin install --batch http://es-learn-to-rank.labs.o19s.com/ltr-plugin-v1.5.1-es7.9.2.zip

# Copying the Elasticsearch configuration files.
COPY config/elasticsearch.yml /usr/share/elasticsearch/config/

# Settting the correct permissions.
RUN chown -R elasticsearch:elasticsearch /usr/share/elasticsearch/config/

# Exposing the Elasticsearch port.
EXPOSE 9200 9300

# Starting Elasticsearch.
CMD ["/usr/share/elasticsearch/bin/elasticsearch"]
