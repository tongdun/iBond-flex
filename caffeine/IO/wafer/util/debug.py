"""
 Created by liwei on 2021/4/8.
"""
import click
import pymysql


class Context:
    def __enter__(self):
        self.db_host = pymysql.connect(
            host="10.58.14.13", port=3307, user="root", passwd="TD@123", database="bond"
        )
        self.db_guest = pymysql.connect(
            host="10.58.14.17", port=3307, user="root", passwd="TD@123", database="bond"
        )
        self.db_arbiter = pymysql.connect(
            host="10.58.14.21", port=3307, user="root", passwd="TD@123", database="bond"
        )

        self.cursor_host = self.db_host.cursor()
        self.cursor_guest = self.db_guest.cursor()
        self.cursor_arbiter = self.db_arbiter.cursor()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db_guest.close()
        self.db_host.close()
        self.db_arbiter.close()


def generate_command(
    project_name, operator_name, debuger, mount_code_base_path, image_version
):
    sql_guest = f"""
    select a.name           as '算子名称',
           a.code           as '算子代码',
           a.operator_class as '算子类',
           concat('/home/tdops/deploy/data/jobconfig/', c.operator_instance_id, '/',
                  c.id)     as '算子配置路径',
           concat('/home/tdops/deploy/data/modelio/', d.uuid, '/', a.code, '_0/')
                            as 'modelio路径',
           concat('/home/tdops/deploy/data/dataio/', d.uuid, '/', a.code, '_0/')
                            as 'dataio路径',
           d.uuid           as '联邦uuid',
           c.*
    from tm_operator_def a,
         tm_operator_instance b,
         tm_operator_job c,
         tm_dag_task d,
         (select dag.id
          from tm_dag dag,
               tm_project pro
          where pro.name = '{project_name}'
            and pro.id = dag.project_id
--             and dag.uuid is not null
          order by dag_task_id desc
          limit 1) as e
    where e.id = d.dag_id
      and c.operator_instance_id = b.id
      and b.operator_code = a.code
      and c.dag_task_id = d.id
      and a.name = '{operator_name}';
    """

    sql_other = f"""
    select a.name           as '算子名称',
           a.code           as '算子代码',
           a.operator_class as '算子类',
           concat('/home/tdops/deploy/data/jobconfig/', c.operator_instance_id, '/',
                  c.id)     as '算子配置路径',
           concat('/home/tdops/deploy/data/modelio/', d.uuid, '/', a.code, '_0/')
                            as 'modelio路径',
           concat('/home/tdops/deploy/data/dataio/', d.uuid, '/', a.code, '_0/')
                            as 'dataio路径',
           d.uuid           as '联邦uuid',
           c.*
    from tm_operator_def a,
         tm_operator_instance b,
         tm_operator_job c,
         tm_dag_task d
    where c.operator_instance_id = b.id
      and b.operator_code = a.code
      and c.dag_task_id = d.id
      and d.uuid = '{{0}}'
      and a.name = '{operator_name}';
    """

    docker_shell = f"""
    docker rm {debuger}-debug -f;
    docker run -d -it --name {debuger}-debug \
    -v {{0}}:/home/tdops/deploy/data/jobconfig \
    -v ~/deploy/data/dataio:/home/tdops/deploy/data/dataio \
    -v ~/deploy/data/dataset:/home/tdops/deploy/data/dataset \
    -v ~/deploy/data/model:/home/tdops/deploy/data/model \
    -v ~/deploy/data/modelio:/home/tdops/deploy/data/modelio \
    -v {mount_code_base_path}/caffeine/fl:/usr/local/lib/python3.6/site-packages/fl \
    -v {mount_code_base_path}/wafer/wafer:/usr/local/lib/python3.6/site-packages/wafer \
    -v {mount_code_base_path}/flex:/usr/local/lib/python3.6/site-packages/flex \
    bond-algo.base:{image_version} bash;
    docker exec -it {debuger}-debug bash;
    """
    return sql_guest, sql_other, docker_shell


@click.command()
@click.option(
    "--image_version",
    "-v",
    prompt="请输入算子容器版本",
    help="算子容器版本",
    default="v2.3.2",
    type=str,
)
@click.option("--code_path", "-p", prompt="请输入映射代码路径", help="映射代码路径", type=str)
@click.option("--debuger", "-d", prompt="请输入开发者昵称", help="开发者昵称", type=str)
@click.option("--operator_name", "-on", prompt="请输入算子名称", help="算子名称", type=str)
@click.option("--project_name", "-pn", prompt="请输入项目名称", help="项目名称", type=str)
def main(project_name, operator_name, debuger, code_path, image_version):
    sql_guest, sql_other, docker_shell = generate_command(
        project_name, operator_name, debuger, code_path, image_version
    )

    with Context() as c:
        c.cursor_guest.execute(sql_guest)
        data_guest = c.cursor_guest.fetchone()
        fed_uuid = data_guest[6]

        c.cursor_host.execute(sql_other.format(fed_uuid))
        data_host = c.cursor_host.fetchone()

        c.cursor_arbiter.execute(sql_other.format(fed_uuid))
        data_arbiter = c.cursor_arbiter.fetchone()

        print("*" * 10, "guest", "*" * 10)
        print(docker_shell.format(data_guest[3]))
        print(data_guest[4])
        print(data_guest[5])

        print("*" * 10, "host", "*" * 10)
        print(docker_shell.format(data_host[3]))
        print(data_host[4])
        print(data_host[5])

        print("*" * 10, "arbiter", "*" * 10)
        print(docker_shell.format(data_arbiter[3]))
        print(data_arbiter[4])
        print(data_arbiter[5])


if __name__ == "__main__":
    # python ./debug.py -v v2.3.2 -p /home/tdops/wei.li -d liwei -on 特征处理 -pn liwei
    main()
