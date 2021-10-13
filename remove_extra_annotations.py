to_remove = ['01', '03', '06', '08', '10', '11']
base = '3characters1car_Insert_'


def remove_extra(fname):
    lines = open(fname, 'r').readlines()
    filtered = []
    for line in lines:
        remove = False
        for r in to_remove:
            if remove:
                break
            if base + r in line:
                remove = True
        if not remove:
            filtered.append(line)

    with open(fname, 'w') as f:
        f.writelines(filtered)


if __name__ == '__main__':
    remove_extra('improved_both.csv')
    remove_extra('improved_masked.csv')
