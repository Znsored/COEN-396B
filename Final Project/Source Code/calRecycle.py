import matplotlib.pyplot as plt
import numpy as np
import mpld3
from mpld3._server import serve
import pandas as pd
from sklearn.neural_network import MLPRegressor

material_order = ['Organics', 'Paper', 'Inerts and Other', 'Plastic', 'Metal', 'Special Waste', 'Miscellaneous', 'Glass', 'Electronics', 'Household Hazardous Waste']

years = [2021, 2018, 2014, 2008, 2003, 1999]
types = ["Statewide_Disposal", "Self_Hauled", "Franchised_Residential", "Franchised_Commercial"]

html_tonnage = []
html_percentages = []

for t in types:
    aggregate_year_data = []
    valid_years = []

    for year in years:
        try:
            aggregate_year_data.append(pd.read_csv("data/" + str(year) + "-" + str(t) + ".csv"))
            valid_years.append(year)
        except:
            print("Missing expected data for " + str(year) + " " + str(t))

    counter = 0
    for data in aggregate_year_data:
        data = data.set_index("Material")
        aggregate_year_data[counter] = data.loc[material_order]
        counter += 1

    tonnage_regressions = []
    percentage_regressions = []
    prediction_years = np.array([[2032], [2028], [2024]])
    tonnage_predictions = []    # [[multiple years per material]]
    percentage_predictions = []
    for mat in material_order:
        Y_tonnage = []
        Y_percentage = []
        X_valid = []
        counter = 0
        for x in valid_years:
            if not np.isnan(aggregate_year_data[counter]["Estimated Tonnage"][mat]):
                Y_tonnage.append(aggregate_year_data[counter]["Estimated Tonnage"][mat])
                Y_percentage.append(aggregate_year_data[counter]["Estimated Proportion (%)"][mat])
                X_valid.append(x)
            counter += 1
        X_valid = np.array(X_valid).reshape((len(X_valid), 1))

        # normalize X values
        X_min = np.amin(X_valid)
        X_norm = np.subtract(X_valid, X_min)
        X_norm_max = np.amax(X_norm) / 10
        X_norm = np.divide(X_norm, X_norm_max)

        # normalize Y_tonnage values
        Y_tonnage_min = np.amin(Y_tonnage)
        Y_tonnage_norm = np.subtract(Y_tonnage, Y_tonnage_min)
        Y_tonnage_norm_max = np.amax(Y_tonnage_norm) / 10
        Y_tonnage_norm = np.divide(Y_tonnage, Y_tonnage_norm_max)

        # normalize Y_percentage values
        Y_percentage_min = np.amin(Y_percentage)
        Y_percentage_norm = np.subtract(Y_percentage, Y_percentage_min)
        Y_percentage_norm_max = np.amax(Y_percentage_norm) / 10
        Y_percentage_norm = np.divide(Y_percentage, Y_percentage_norm_max)

        # train
        tonnage_regressions.append(MLPRegressor(random_state=1,
                                                max_iter=2000,
                                                hidden_layer_sizes=(100, 50, 10)).fit(X_norm, Y_tonnage_norm))
        percentage_regressions.append(MLPRegressor(random_state=1,
                                                   max_iter=2000,
                                                   hidden_layer_sizes=(100, 50, 10)).fit(X_norm, Y_percentage_norm))
        print(t + " " + mat + " tonnage regression score: " +
              str(tonnage_regressions[-1].score(X_norm, Y_tonnage_norm)))
        print(t + " " + mat + " percentage regression score: " +
              str(percentage_regressions[-1].score(X_norm, Y_percentage_norm)))

        # normalize prediction years
        pred_X = prediction_years
        pred_X = np.subtract(pred_X, X_min)
        pred_X = np.divide(pred_X, X_norm_max)

        # predict
        tonnage_preds = tonnage_regressions[-1].predict(pred_X)
        percentage_preds = percentage_regressions[-1].predict(pred_X)

        # de-normalize tonnage predictions
        tonnage_preds = np.multiply(tonnage_preds, Y_tonnage_norm_max)
        tonnage_preds = np.add(tonnage_preds, Y_tonnage_min)
        tonnage_preds = np.rint(tonnage_preds)
        tonnage_preds = [int(entry) for entry in tonnage_preds]
        tonnage_predictions.append(tonnage_preds)

        # de-normalize percentage predictions
        percentage_preds = np.multiply(percentage_preds, Y_percentage_norm_max)
        percentage_preds = np.add(percentage_preds, Y_percentage_min)

        percentage_predictions.append(percentage_preds)

    new_tonnage_by_year = np.array(tonnage_predictions).T
    new_percentage_by_year = np.array(percentage_predictions).T

    # convert to percentages
    for i, entry in enumerate(new_percentage_by_year):
        new_percentage_by_year[i] = entry * (100 / np.sum(entry))

    n_entries = len(material_order)
    x_indices = range(0, n_entries)
    colors = ["purple", "orange", "blue", "red", "green"]

    # plot tonnages
    fig, ax = plt.subplots(subplot_kw=dict(facecolor='#EEEEEE'))
    labels = ["2032", "2028", "2024"]
    for valid_year in valid_years:
        labels.append(str(valid_year))

    ax.grid(color='white', linestyle='--', alpha=0.3)
    ax.set_title("California " + str(t) + " Waste by Year (Tonnage)", size=20)
    plt.xlabel("Waste Type (Enumerated)")
    plt.ylabel("Tonnage (Millions)")

    layers = []

    counter = 0
    for data in new_tonnage_by_year:
        layers.append(plt.fill_between(x_indices, list(data/1000000),
                                       alpha=0.4,
                                       color=colors[counter % 5]))
        counter += 1
    for data in aggregate_year_data:
        layers.append(plt.fill_between(x_indices, list(data["Estimated Tonnage"]/1000000),
                                       alpha=0.5,
                                       color=colors[counter % 5]))
        counter += 1

    interactive_legend = mpld3.plugins.InteractiveLegendPlugin(layers, labels)
    mpld3.plugins.connect(fig, interactive_legend)

    for year in aggregate_year_data:
        scatter = ax.scatter(x_indices, list(year["Estimated Tonnage"]/1000000), s=10, alpha=0)
        labs = []
        for data_point in year["Estimated Tonnage"]:
            if not np.isnan(data_point):
                labs.append(f'{(data_point):,}')
        tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=labs)
        mpld3.plugins.connect(fig, tooltip)

    for year in new_tonnage_by_year:
        scatter = ax.scatter(x_indices, list(year/1000000), s=10, alpha=0)
        tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=[f'{value:,}' for value in list(year)])
        mpld3.plugins.connect(fig, tooltip)

    html_tonnage.append(mpld3.fig_to_html(fig))

    # plot percentages
    fig, ax = plt.subplots(subplot_kw=dict(facecolor='#EEEEEE'))
    ax.grid(color='white', linestyle='--', alpha=0.3)
    ax.set_title("California " + str(t) + " Waste by Year (Percentage)", size=20)
    plt.xlabel("Year")
    plt.ylabel("Percentage")

    counter = 0
    aggregate_year_data.reverse()
    data_years = valid_years
    data_years.reverse()
    predicted_years = [2024, 2028, 2032]
    percentage_years = data_years + predicted_years
    percentages_by_year = []
    for data in aggregate_year_data:
        data = data.fillna(0)
        percentages_by_year.append(list(data["Estimated Proportion (%)"]))
    for data in new_percentage_by_year:
        percentages_by_year.append(data)
    percentages_by_year = np.array(percentages_by_year).T
    percentages_by_year[np.isnan(percentages_by_year)] = 0

    plt.stackplot(percentage_years, percentages_by_year,
                  labels=['Organics',
                          'Paper',
                          'Inerts',
                          'Plastic',
                          'Metal',
                          'Special',
                          'Misce',
                          'Glass',
                          'Electronics',
                          'HHZ'],
                  colors=colors)

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(reversed(handles), reversed(labels), bbox_to_anchor=(1.22, 0.75))

    html_percentages.append(mpld3.fig_to_html(fig))

radio_list_html = '	<div>\n' \
                  '<label>\n' \
                  '<input type="radio" name="colorRadio" checked="checked"\n' \
                  'value="SD"> Statewide Disposal</label>\n' \
                  '<label>\n' \
                  '<input type="radio" name="colorRadio" value="SH"> Self Hauled</label>\n' \
                  '<label>\n' \
                  '<input type="radio" name="colorRadio" value="FR"> Franchised Residential</label>\n' \
                  '<label>\n' \
                  '<input type="radio" name="colorRadio" value="FC"> Franchised Commercial</label>\n' \
                  '</div>'

radio_list_script = '<script src="https://code.jquery.com/jquery-1.12.4.min.js">\n' \
                    '</script>\n' \
                    '<script type="text/javascript">\n' \
                    '$(document).ready(function() {\n' \
                        '$(\'input[type="radio"]\').click(function() {\n' \
                            'var inputValue = $(this).attr("value");\n' \
                            'var targetBox = $("." + inputValue);\n' \
                            '$(".selectt").not(targetBox).hide();\n' \
                            '$(targetBox).show();\n' \
                            '});\n' \
                        '});\n' \
                    '</script>'

enumerated_waste_types = '<div style="width: 50%; position: absolute; left: 700px;">'
for i, mat in enumerate(material_order):
    enumerated_waste_types += '<div>'
    enumerated_waste_types += str(i)
    enumerated_waste_types += ': '
    enumerated_waste_types += str(mat)
    enumerated_waste_types += '</div>'
enumerated_waste_types += '</div>'

serve(radio_list_html +
      '<div class="SD selectt" style="width: 100%;">' +
      '<div style="width: 50%; float: left;">' +
      html_tonnage[0] + html_percentages[0] +
      '</div>' +
      enumerated_waste_types +
      '</div>'
      '<div class="SH selectt" style="width: 100%; display: none;">' + '<div style="width: 50%; float: left;">' +
      html_tonnage[1] + html_percentages[1] +
      '</div>' +
      enumerated_waste_types +
      '</div>'
      '<div class="FR selectt" style="width: 100%; display: none;">' + '<div style="width: 50%; float: left;">' +
      html_tonnage[2] + html_percentages[2] +
      '</div>' +
      enumerated_waste_types +
      '</div>'
      '<div class="FC selectt" style="width: 100%; display: none;">' + '<div style="width: 50%; float: left;">' +
      html_tonnage[3] + html_percentages[3] +
      '</div>' +
      enumerated_waste_types +
      radio_list_script,
      ip="192.168.0.20")