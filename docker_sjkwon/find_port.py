#!/usr/bin/python3.6
import json
import os
import subprocess


def get_container_id():
    output = subprocess.check_output("docker ps -a --format {{.ID}}", shell=True)  
    return output.decode('utf-8').split('\n')[:-1]


def inspect_id(value):
    output = subprocess.check_output("docker inspect {}".format(value), shell=True)
    result = output.decode('utf-8').replace('\n', '')[1:-1]
    return result


def check_ports(resource, used_ports):
    for port in used_ports:
        if port in resource:
            resource.remove(port)
    return resource


def print_ports_list(ports):
    tmp = []
    for i, p in enumerate(ports):
        tmp.append(str(p))
        if ((i+1) % 5 == 0) or (len(ports) == i + 1):
            print('\t{}'.format('\t'.join(tmp)))
            tmp = []


def main():
    range_ports = list(range(8801, 8900))
    ids = get_container_id()

    used_ports = {}
    for _id in ids:
        detailed_str = inspect_id(_id)
        detailed_dict = json.loads(detailed_str)
        ports = detailed_dict['HostConfig']['PortBindings']
        if type(ports) == dict:
            if len(ports) > 0:
                _container_name = detailed_dict['Name']
                for k, v in ports.items():
#                     import pdb;pdb.set_trace()
                    try:
                        _port = int(v[0]['HostPort'])
                    except:
                        continue
                    if _port in range_ports:
                        if _port not in used_ports:
                            used_ports[_port] = [_container_name]
                        else:
                            used_ports[_port] += [_container_name]

    unused_ports = check_ports(range_ports, used_ports.keys())

    _used_ports_list = list(used_ports.keys())
    _used_ports_list.sort()
    unused_ports.sort()

    for port in _used_ports_list:
        print('{}\t{}'.format(port, used_ports[port]))

    print('\nUsing ports: {}'.format(len(used_ports)))
    print_ports_list(_used_ports_list)

    print('\nUnused ports: {}'.format(len(unused_ports)))
    print_ports_list(unused_ports)


if __name__== '__main__':
    main()
