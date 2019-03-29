import pika

connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channelin = connection.channel()


channelin.exchange_declare(exchange='chat')

chat = 'boo'

channelin.basic_publish(exchange='chat',
                      routing_key='chatin',
                      body=chat)
print(" [x] Sent '{0}'".format(chat))
connection.close()

