/**
 * Created by 志俊 on 2016/5/29 0029.
 */
function Set_chart()
{

    var randomColorFactor = function() {
        return Math.round(Math.random() * 255);
    };
    var randomColor = function(opacity) {
        return 'rgba(' + randomColorFactor() + ',' + randomColorFactor() + ',' + randomColorFactor() + ',' + (opacity || '.3') + ')';
    };
    $.ajax({
        type:"post",
        url:$app+"Admin/ShowExarticleamountstatInfo",
        textType:"json",
        success:function(data){
            var config = {
                type: 'line',
                data: {
                    labels: data['labels'],
                    datasets:  data['datasets']
                },
                options: {
                    scaleOverride : true,
                    scaleShowLabels : true,
                    responsive: true,
                    title:{
                        display:true,
                        text:'博文和爬虫数据统计'
                    },
                    tooltips: {
                        mode: 'label',
                        callbacks: {
                        }
                    },
                    hover: {
                        mode: 'dataset'
                    },
                    scales: {
                        xAxes: [{
                            display: true,
                            scaleLabel: {
                                show: true,
                                labelString: 'Day'
                            }
                        }],
                        yAxes: [{
                            display: true,
                            scaleLabel: {
                                show: true,
                                labelString: 'Number'
                            },
                            ticks: {
                                suggestedMin: 0,
                                suggestedMax: data['suggestedMax'],
                            }
                        }]
                    }
                }
            };
            var i_num=0;
            $.each(config.data.datasets, function(i, dataset) {
                dataset.borderColor = randomColor(0.4);
                backgroundColor=randomColor(0.7);
                dataset.backgroundColor = backgroundColor;
                dataset.pointBorderColor = randomColor(0.7);
                dataset.pointBackgroundColor = randomColor(0.5);
                dataset.pointBorderWidth = 1;
                $("#li_info_"+i_num++).css("color",backgroundColor);
            });

            var ctx = $("#myChart").get(0).getContext("2d");
            var myNewChart = new Chart(ctx,config);
        }
    });
}


$(function () {
    if($("#myChart"))
    {
        Set_chart();
    }
});