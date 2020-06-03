/**
 * Created by 志俊 on 2016/5/31.
 */
$(function () {
    var $modal = $('#am-alert');
    $(".city_edit").click(function () {
        $id=$(this).attr("data-id");
        $('#am-edit-confirm').modal({
            onConfirm: function(options) {
                $("#city_edit_"+$id).hide();
                $("#city_sumbit_"+$id).show();
                $("#city_exit_"+$id).show();
                $("#city_delete_"+$id).hide();
                $("#cities_item_name_"+$id).hide();
                $("#cities_input_name_"+$id).show();
            },
            onCancel: function() {
                alert('算求，不弄了');
            }
        });
    });
    $(".city_exit").click(function () {
        $id=$(this).attr("data-id");
        $('#am-exit-confirm').modal({
            onConfirm: function(options) {
                $("#city_edit_"+$id).show();
                $("#city_sumbit_"+$id).hide();
                $("#city_exit_"+$id).hide();
                $("#city_delete_"+$id).show();
                $("#cities_item_name_"+$id).show();
                $("#cities_input_name_"+$id).hide();
            },
            onCancel: function() {
                alert('继续编辑');
            }
        });
    });
    $(".city_sumbit").click(function () {
        $id=$(this).attr("data-id");
        $name=$("#cities_input_name_"+$id).val();
        $.ajax({
            url:$app+"Admin/city_edit",
            type:"post",
            data:{id:$id,name:$name},
            dataType:"json",
            success:function (data) {
                $modal.on('open.modal.amui', function(){
                    $("#am_alert_warning").html(data.info);
                });
                $modal.modal();
                $("#city_edit_"+$id).show();
                $("#city_sumbit_"+$id).hide();
                $("#city_exit_"+$id).hide();
                $("#city_delete_"+$id).show();
                $("#cities_item_name_"+$id).html($("#cities_input_name_"+$id).val());
                $("#cities_item_name_"+$id).show();
                $("#cities_input_name_"+$id).hide();
            }
        });
    });
    $(".city_delete").click(function () {
        $id=$(this).attr("data-id");
        $('#am-delete-confirm').modal({
            onConfirm: function(options) {
                $.ajax({
                    url:$app+"Admin/city_delete",
                    type:"post",
                    data:{id:$id},
                    dataType:"json",
                    success:function (data) {
                        if (data.status == 1) {
                            cities_info="#cities_info_"+$id;
                            $(cities_info).remove();
                        }
                        else {
                            var $modal = $('#am-alert');
                            $modal.modal();
                            $modal.on('open.modal.amui', function(){
                                $("#am_alert_warning").html(data.info)
                            });
                        }
                    }
                })
            },
            // closeOnConfirm: false,
            onCancel: function() {
                alert('算求，不弄了');
            }
        });
    });
    $("#model_cities_sumbit").click(function () {
        var $name=$('#model_cities_name').val();
        $.ajax({
            url:$app+"Admin/city_add",
            type:"post",
            data:{name:$name},
            dataType:"json",
            success:function (data) {
                if (data.status == 1) {
                    var $modal = $('#add_city_model');
                    $modal.modal('close');
                    window.location.href=data.url;
                }
                else {
                    $("#model_warning").html(data.info);
                }
            }
        });
    })
});

