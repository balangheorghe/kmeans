#! python3
import os
import numpy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def load_file(path):
    handler = open(path, "r")
    content = handler.readlines()
    handler.close()
    output = []
    for line in content:
        line = line.split(" ")
        x = float(line[0].strip())
        y = float(line[1].strip())
        output.append((x, y))
    return output


def combine_jscore_iter(j_score_list, iter_list):
    plt.cla()
    flg = plt.figure()
    x = [0]
    iter = 0
    for i in range(0, len(j_score_list) - 1):
        x.append(float(i / 2 + 1))
    # x = [item[0] for item in j_score_list]
    # y = [item[1] for item in j_score_list]
    plt.plot(x, j_score_list, color='b', marker='.')
    tt = list(x)
    x = []
    for i in range(0, len(iter_list)):
        x.append(i)
    plt.xticks(list(set(x + tt)))
    inter_score_list = [x * 50 for x in iter_list]
    # x = [item[0] for item in inter_score_list]
    # y = [item[1] for item in inter_score_list]
    plt.plot(x, inter_score_list, color='r', marker='.')

    plt.title('jscore variation (blue) and inter-clustere variation (red)')
    plt.ylabel('jscore (blue) and iter-cluster * 50 (red)')
    plt.xlabel('iteration')
    flg.savefig("combined_variation")


def generate_jscore_graphic(j_score_list):
    plt.cla()
    flg = plt.figure()
    x = [0]
    iter = 0
    for i in range(0, len(j_score_list)-1):
        x.append(float(i/2 + 1))
    plt.xticks(x)
    plt.yticks(j_score_list)
    # x = [item[0] for item in j_score_list]
    # y = [item[1] for item in j_score_list]
    plt.plot(x, j_score_list, color='b', marker='.')
    for i in range(0, len(x)):
        if i == 0:
            to_annotate = "C0,u0"
            to_xytext = (-20, 20)
        else:
            if i%2 == 1:
                to_annotate = "C{tt},u{t}".format(tt=int(i/2),t=int(i/2)+1)
                to_xytext = (-10, 20)
            else:
                to_annotate = "C{t},u{t}".format(t=int(i/2))
                to_xytext = (0, 30)

        plt.annotate(
            to_annotate,
            xy=(x[i], j_score_list[i]), xytext=to_xytext,
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    plt.title('jscore variation')
    plt.ylabel('score')
    plt.xlabel('iteration')
    flg.savefig("jscore_variation")


def generate_inter_score_graphic(inter_score_list):
    plt.cla()
    flg = plt.figure()
    x = []
    for i in range(0, len(inter_score_list)):
        x.append(i)
    plt.xticks(x)
    plt.yticks(inter_score_list)
    # x = [item[0] for item in inter_score_list]
    # y = [item[1] for item in inter_score_list]
    plt.plot(x, inter_score_list, color='b', marker='.')
    for i in range(0, len(x)):
        plt.annotate(
            "{}".format(i),
            xy=(x[i], inter_score_list[i]), xytext=(-20, 20),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=1),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    plt.title('inter-clustere variation')
    plt.ylabel('score')
    plt.xlabel('iteration')
    flg.savefig("inter_clustere_variation")


def ploting(clusters, new_centroids, step):
    plt.axis([0, 1, 0, 1])
    index = 0
    # color_centroid = ['m', 'y', 'k']
    color_cluster = ['b', 'r', 'g']
    color_centroid = ['b', 'r', 'g']
    color_old_centroid = ['#33D4FF', '#FF33E6', '#33FF6B']
    plt.cla()
    flg = plt.figure()
    if new_centroids:
        for centroid in new_centroids:
            plt.plot(centroid[0], centroid[1], '{}x'.format(color_centroid[index]))
            plt.annotate(
                "new_centroid".format(index + 1),
                xy=(centroid[0], centroid[1]), xytext=(-20, 20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            index += 1
    index = 0
    for cluster in clusters:
        plt.plot(cluster[0], cluster[1], color=color_old_centroid[index], marker='x')
        if new_centroids is not None:
            if not new_centroids[index][0] == cluster[0] or not new_centroids[index][1] == cluster[1]:
                plt.annotate(
                    "old_centroid".format(index + 1),
                    xy=(cluster[0], cluster[1]), xytext=(-20, 20),
                    textcoords='offset points', ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        else:
            plt.annotate(
                "centroid".format(index + 1),
                xy=(cluster[0], cluster[1]), xytext=(-20, 20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        if new_centroids:
            x = [cluster[0], new_centroids[index][0]]
            y = [cluster[1], new_centroids[index][1]]
            plt.plot(x, y, linestyle='--', linewidth=0.5)
            pass
            # plt.plot(cluster, , marker='--')
        x = [point[0] for point in clusters[cluster]]
        y = [point[1] for point in clusters[cluster]]
        plt.plot(x, y, '{}o'.format(color_cluster[index]))
        if new_centroids:
            for point in clusters[cluster]:
                x = [new_centroids[index][0], point[0]]
                y = [new_centroids[index][1], point[1]]
                plt.plot(x, y, linestyle='--', linewidth=0.5)
        index += 1
    # plt.show()
    # plt._imsave("iteratie_{}.jpg".format(step))
    # plt.imsave("iteratie_{}.jpg".format(step))
    flg.savefig("iteratie_{}.jpg".format(step))


def computing_j(clusters, j_len):
    # j_coeficient = []
    # for i in range(0, j_len):
    #     j_coeficient.append(0)
    j_coeficient = 0
    for cluster in clusters:
        for point in clusters[cluster]:
            x = []
            for i in range(0, len(point)):
                #x.append((point[i] - cluster[i]) * (point[i] - cluster[i]))
                x.append(point[i] - cluster[i])
            custom_sum = 0
            for t in x:
                custom_sum += t*t
            # j_coeficient = [j_coeficient[i] + x[i] for i in range(0, len(point))]
            j_coeficient += custom_sum
    return j_coeficient

def computing_j_intermediar(clusters, new_centroids, j_len):
    # j_coeficient = []
    j_coeficient = 0
    k = 0
    # for i in range(0, j_len):
    #     j_coeficient.append(0)
    for cluster in clusters:
        for point in clusters[cluster]:
            x = []
            for i in range(0, len(point)):
                x.append(point[i] - new_centroids[k][i])
            custom_sum = 0
            for t in x:
                custom_sum += t * t
            # j_coeficient = [j_coeficient[i] + x[i] for i in range(0, len(point))]
            j_coeficient += custom_sum
        k += 1
    return j_coeficient

def computing_interclustere(clusters, mean_point, j_len, n):
    inter_ = 0
    x = []
    for i in range(0, j_len):
    #     inter_.append(0)
         x.append(0)
    for cluster in clusters:
        n_tr = len(clusters[cluster])
        pondere = n_tr * 1.0 / n
        for i in range(0, len(cluster)):
            x[i] = (cluster[i] - mean_point[i])
        custom_sum = 0
        for t in x:
            custom_sum += t * t
        inter_ += pondere * custom_sum
    return inter_


def kmeans(iteration, points, centroids, clusters, j_score_list, mean_point, inter_score_list):
    handle = open("iteratie_{}.date".format(iteration), "w")
    if iteration == 0:
        for point in points:
            distances = []
            for centroid in centroids:
                distance = float("{0:.4f}".format(numpy.sqrt((centroid[0] - point[0]) * (centroid[0] - point[0]) + \
                                      (centroid[1] - point[1]) * (centroid[1] - point[1]))))
                distances.append(distance)
            clusters[centroids[numpy.argmin(distances)]].append(point)
        ploting(clusters, None, iteration)
    else:
        old_centroids = list(centroids)
        for i in range(0, len(centroids)):
            if len(clusters[centroids[i]]) != 0:
                # clusters[centroids[i]].append(centroids[i])
                centroids[i] = tuple([float("{0:.4f}".format(x)) for x in\
                                      numpy.sum(clusters[centroids[i]], axis=0)/ float(len(clusters[centroids[i]]))])
        new_centroids = list(centroids)
        # print("Old centroids: {}".format(old_centroids))
        # handle.write("Old centroids: {}\n".format(old_centroids))
        # print("New centroids: {}".format(new_centroids))
        # handle.write("New centroids: {}\n".format(new_centroids))
        ploting(clusters, new_centroids, iteration)
        if old_centroids == new_centroids:
            return clusters
        j_intermediar = computing_j_intermediar(clusters, new_centroids, len(points[0]))
        print("j_score_intermediat: {}\n".format(j_intermediar))
        handle.write("j_score_intermediat: {}\n".format(j_intermediar))
        j_score_list.append(j_intermediar)
        clusters = dict()
        for i in range(0, len(centroids)):
            clusters[centroids[i]] = []
        for point in points:
            distances = []
            for centroid in centroids:
                distance = float("{0:.4f}".format(numpy.sqrt((centroid[0] - point[0]) * (centroid[0] - point[0]) + \
                                      (centroid[1] - point[1]) * (centroid[1] - point[1]))))
                distances.append(distance)
            clusters[centroids[numpy.argmin(distances)]].append(point)
    j_score = computing_j(clusters, len(points[0]))
    inter_clustere = computing_interclustere(clusters, mean_point, len(points[0]), len(points))
    print("KMEANS: \nIteration: {}".format(iteration))
    handle.write("KMEANS: \nIteration: {}\n".format(iteration))
    print("j_score: {}\n".format(j_score))
    print("inter_clustere: {}\n".format(inter_clustere))
    handle.write("j_score: {}\n".format(j_score))
    handle.write("inter_clustere: {}\n".format(inter_clustere))
    j_score_list.append(j_score)
    inter_score_list.append(inter_clustere)
    print("\nPoints: {}\nCentroids: {}\nClusters: {}\n".format(points, centroids, clusters))
    handle.write("Points: {}\nCentroids: {}\nClusters: {}\n".format(points, centroids, clusters))
    iteration += 1
    return kmeans(iteration, points, centroids, clusters, j_score_list, mean_point, inter_score_list)


if __name__ == '__main__':
    points = list(load_file("points"))
    centroids = list(load_file("centroid"))
    iteration = 0
    clusters = dict()
    for centroid in centroids:
        clusters[centroid] = []
    j_score_list = []
    inter_score_list = []
    mean_point = []
    for i in range(len(points[0])):
        sum_tre = 0
        for item in points:
            sum_tre += item[i]
        mean_point.append(sum_tre * 1.0 / len(points))
    new_clutser = kmeans(iteration, points, centroids, clusters, j_score_list, mean_point, inter_score_list)

    generate_jscore_graphic(j_score_list)
    print(j_score_list)

    generate_inter_score_graphic(inter_score_list)
    print(inter_score_list)

    combine_jscore_iter(j_score_list,inter_score_list)