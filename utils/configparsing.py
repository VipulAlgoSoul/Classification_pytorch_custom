import configparser

config = configparser.ConfigParser()

def create_config(litudi):
    '''input list of tuple dicts'''

    for i in litudi:
        config[i[0]]=i[1]


    with open("config.ini",'w') as configfile:
        config.write(configfile)


m=[("DEFAULT",{}),("w_and_b",{"Authkey":"xxxxxxxxxxxxxxxxxxxxxxx"})]
create_config(m)