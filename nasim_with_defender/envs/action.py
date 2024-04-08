import math
import numpy as np
from gymnasium import spaces

from nasim_with_defender.envs.utils import AccessLevel


def load_action_list(scenario):
    action_list = []
    for address in scenario.address_space:
        action_list.append(
            ServiceScan(address, scenario.service_scan_cost)
        )
        action_list.append(
            OSScan(address, scenario.os_scan_cost)
        )
        action_list.append(
            SubnetScan(address, scenario.subnet_scan_cost)
        )
        action_list.append(
            ProcessScan(address, scenario.process_scan_cost)
        )
        for e_name, e_def in scenario.exploits.items():
            exploit = Exploit(e_name, address, **e_def)
            action_list.append(exploit)
        for pe_name, pe_def in scenario.privescs.items():
            privesc = PrivilegeEscalation(pe_name, address, **pe_def)
            action_list.append(privesc)
    return action_list

def load_defender_action_list(scenario):        # defender
    action_list = []
    for address in scenario.defender:
        action_list.append(
            Change_Os(address, scenario.change_os_cost)
        )
        action_list.append(
            Change_Firewall(address, scenario.chagne_firewall_cost)
        )
        action_list.append(
            Stop_Service(address, scenario.stop_service_cost)
        )
        action_list.append(
            Stop_Process(address, scenario.stop_processes_cost)
        )
    return action_list

class Action:
    def __init__(self,
                 name,
                 target,
                 cost,
                 prob=1.0,
                 req_access=AccessLevel.USER,
                 **kwargs):
        assert 0 <= prob <= 1.0
        self.name = name
        self.target = target
        self.cost = cost
        self.prob = prob
        self.req_access = req_access

    def is_exploit(self):
        return isinstance(self, Exploit)

    def is_privilege_escalation(self):
        return isinstance(self, PrivilegeEscalation)

    def is_scan(self):
        return isinstance(self, (ServiceScan, OSScan, SubnetScan, ProcessScan))

    def is_remote(self):
        return isinstance(self, (ServiceScan, OSScan, Exploit))

    def is_service_scan(self):
        return isinstance(self, ServiceScan)

    def is_os_scan(self):
        return isinstance(self, OSScan)

    def is_subnet_scan(self):
        return isinstance(self, SubnetScan)

    def is_process_scan(self):
        return isinstance(self, ProcessScan)

    def is_noop(self):
        return isinstance(self, NoOp)

    def __str__(self):
        return (f"{self.__class__.__name__}: "
                f"target={self.target}, "
                f"cost={self.cost:.2f}, "
                f"prob={self.prob:.2f}, "
                f"req_access={self.req_access}")

    def __hash__(self):
        return hash(self.__str__())

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, type(self)):
            return False
        if self.target != other.target:
            return False
        if not (math.isclose(self.cost, other.cost)
                and math.isclose(self.prob, other.prob)):
            return False
        return self.req_access == other.req_access

class Action_Defender:      # defender
    def __init__(self,
                 name,
                 target,
                 cost,
                 prob=1.0,
                 req_access=AccessLevel.USER,
                 **kwargs):
        assert 0 <= prob <= 1.0
        self.name = name
        self.target = target
        self.cost = cost
        self.prob = prob
        self.req_access = req_access

    def is_change_os(self):
        return isinstance(self, Change_Os)

    def is_change_firewall(self):
        return isinstance(self, Change_Firewall)

    def is_stop_process(self):
        return isinstance(self, Stop_Process)

    def is_stop_service(self):
        return isinstance(self, Stop_Service)

    def __str__(self):
        return (f"{self.__class__.__name__}: "
                f"target={self.target}, "
                f"cost={self.cost:.2f}, "
                f"prob={self.prob:.2f}, "
                f"req_access={self.req_access}")

    def __hash__(self):
        return hash(self.__str__())

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, type(self)):
            return False
        if self.target != other.target:
            return False
        if not (math.isclose(self.cost, other.cost)
                and math.isclose(self.prob, other.prob)):
            return False
        return self.req_access == other.req_access

class Exploit(Action):
    def __init__(self,
                 name,
                 target,
                 cost,
                 service,
                 os=None,
                 access=0,
                 prob=1.0,
                 req_access=AccessLevel.USER,
                 **kwargs):
        super().__init__(name=name,
                         target=target,
                         cost=cost,
                         prob=prob,
                         req_access=req_access)
        self.os = os
        self.service = service
        self.access = access

    def __str__(self):
        return (f"{super().__str__()}, os={self.os}, "
                f"service={self.service}, access={self.access}")

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        return self.service == other.service \
            and self.os == other.os \
            and self.access == other.access


class PrivilegeEscalation(Action):
    def __init__(self,
                 name,
                 target,
                 cost,
                 access,
                 process=None,
                 os=None,
                 prob=1.0,
                 req_access=AccessLevel.USER,
                 **kwargs):

        super().__init__(name=name,
                         target=target,
                         cost=cost,
                         prob=prob,
                         req_access=req_access)
        self.access = access
        self.os = os
        self.process = process

    def __str__(self):
        return (f"{super().__str__()}, os={self.os}, "
                f"process={self.process}, access={self.access}")

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        return self.process == other.process \
            and self.os == other.os \
            and self.access == other.access


class ServiceScan(Action):
    def __init__(self,
                 target,
                 cost,
                 prob=1.0,
                 req_access=AccessLevel.USER,
                 **kwargs):
        super().__init__("service_scan",
                         target=target,
                         cost=cost,
                         prob=prob,
                         req_access=req_access,
                         **kwargs)


class OSScan(Action):
    def __init__(self,
                 target,
                 cost,
                 prob=1.0,
                 req_access=AccessLevel.USER,
                 **kwargs):
        super().__init__("os_scan",
                         target=target,
                         cost=cost,
                         prob=prob,
                         req_access=req_access,
                         **kwargs)


class SubnetScan(Action):
    def __init__(self,
                 target,
                 cost,
                 prob=1.0,
                 req_access=AccessLevel.USER,
                 **kwargs):
        super().__init__("subnet_scan",
                         target=target,
                         cost=cost,
                         prob=prob,
                         req_access=req_access,
                         **kwargs)


class ProcessScan(Action):
    def __init__(self,
                 target,
                 cost,
                 prob=1.0,
                 req_access=AccessLevel.USER,
                 **kwargs):
        super().__init__("process_scan",
                         target=target,
                         cost=cost,
                         prob=prob,
                         req_access=req_access,
                         **kwargs)


class NoOp(Action):
    def __init__(self, *args, **kwargs):
        super().__init__(name="noop",
                         target=(1, 0),
                         cost=0,
                         prob=1.0,
                         req_access=AccessLevel.NONE)


#defender action
class Change_Os:
    def __init__(self,
                 target,
                 cost,
                 prob=1.0,
                 req_access=AccessLevel.USER,
                 **kwargs):
        super().__init__(name="change_os",
                         target=target,
                         cost=0,
                         prob=1.0,
                         req_access=AccessLevel.NONE)

class Change_Firewall:
    def __init__(self,
                 target,
                 cost,
                 prob=1.0,
                 req_access=AccessLevel.USER,
                 **kwargs):
        super().__init__(name="change_firewall",
                         target=target,
                         cost=0,
                         prob=1.0,
                         req_access=AccessLevel.NONE)

class Stop_Service:
    def __init__(self,
                 target,
                 cost,
                 prob=1.0,
                 req_access=AccessLevel.USER,
                 **kwargs):
        super().__init__(name="stop_processes",
                         target=target,
                         cost=0,
                         prob=1.0,
                         req_access=AccessLevel.NONE)
class Stop_Process:
    def __init__(self,
                 target,
                 cost,
                 prob=1.0,
                 req_access=AccessLevel.USER,
                 **kwargs):
        super().__init__(name="stop_service",
                         target=target,
                         cost=0,
                         prob=1.0,
                         req_access=AccessLevel.NONE)

class ActionResult:
    def __init__(self,
                 success,
                 value=0.0,
                 services=None,
                 os=None,
                 processes=None,
                 access=None,
                 discovered=None,
                 connection_error=False,
                 permission_error=False,
                 undefined_error=False,
                 newly_discovered=None):

        self.success = success
        self.value = value
        self.services = {} if services is None else services
        self.os = {} if os is None else os
        self.processes = {} if processes is None else processes
        self.access = {} if access is None else access
        self.discovered = {} if discovered is None else discovered
        self.connection_error = connection_error
        self.permission_error = permission_error
        self.undefined_error = undefined_error
        if newly_discovered is not None:
            self.newly_discovered = newly_discovered
        else:
            self.newly_discovered = {}

    def info(self):
        return dict(
            success=self.success,
            value=self.value,
            services=self.services,
            os=self.os,
            processes=self.processes,
            access=self.access,
            discovered=self.discovered,
            connection_error=self.connection_error,
            permission_error=self.permission_error,
            undefined_error=self.undefined_error,
            newly_discovered=self.newly_discovered
        )

    def __str__(self):
        output = ["ActionObservation:"]
        for k, val in self.info().items():
            output.append(f"  {k}={val}")
        return "\n".join(output)


class FlatActionSpace(spaces.Discrete):

    def __init__(self, scenario):
        self.actions = load_action_list(scenario)
        super().__init__(len(self.actions))

    def get_action(self, action_idx):
        assert isinstance(action_idx, int), \
            ("When using flat action space, action must be an integer"
             f" or an Action object: {action_idx} is invalid")
        return self.actions[action_idx]


class FlatDefenderActionSpace(spaces.Discrete):     # defender
    def __init__(self, scenario):
        self.defender_actions = load_defender_action_list(scenario)
        super().__init__(len(self.defender_actions))

    def get_defender_action(self, action_idx):
        assert isinstance(action_idx, int), \
            ("When using flat action space, action must be an integer"
             f" or an Action object: {action_idx} is invalid")
        return self.defender_actions[action_idx]

class ParameterisedActionSpace(spaces.MultiDiscrete):
    action_types = [
        Exploit,
        PrivilegeEscalation,
        ServiceScan,
        OSScan,
        SubnetScan,
        ProcessScan
    ]

    def __init__(self, scenario):
        """
        Parameters
        ----------
        scenario : Scenario
            scenario description
        """
        self.scenario = scenario
        self.actions = load_action_list(scenario)

        nvec = [
            len(self.action_types),
            len(self.scenario.subnets)-1,
            max(self.scenario.subnets),
            self.scenario.num_os+1,
            self.scenario.num_services,
            self.scenario.num_processes
        ]

        super().__init__(nvec)

    def get_action(self, action_vec):
        assert isinstance(action_vec, (list, tuple, np.ndarray)), \
            ("When using parameterised action space, action must be an Action"
             f" object, a list or a numpy array: {action_vec} is invalid")
        a_class = self.action_types[action_vec[0]]
        # need to add one to subnet to account for Internet subnet
        subnet = action_vec[1]+1
        host = action_vec[2] % self.scenario.subnets[subnet]

        target = (subnet, host)

        if a_class not in (Exploit, PrivilegeEscalation):
            # can ignore other action parameters
            kwargs = self._get_scan_action_def(a_class)
            return a_class(target=target, **kwargs)

        os = None if action_vec[3] == 0 else self.scenario.os[action_vec[3]-1]

        if a_class == Exploit:
            # have to make sure it is valid choice
            # and also get constant params (name, cost, prob, access)
            service = self.scenario.services[action_vec[4]]
            a_def = self._get_exploit_def(service, os)
        else:
            # privilege escalation
            # have to make sure it is valid choice
            # and also get constant params (name, cost, prob, access)
            proc = self.scenario.processes[action_vec[5]]
            a_def = self._get_privesc_def(proc, os)

        if a_def is None:
            return NoOp()
        return a_class(target=target, **a_def)

    def _get_scan_action_def(self, a_class):
        """Get the constants for scan actions definitions """
        if a_class == ServiceScan:
            cost = self.scenario.service_scan_cost
        elif a_class == OSScan:
            cost = self.scenario.os_scan_cost
        elif a_class == SubnetScan:
            cost = self.scenario.subnet_scan_cost
        elif a_class == ProcessScan:
            cost = self.scenario.process_scan_cost
        else:
            raise TypeError(f"Not implemented for Action class {a_class}")
        return {"cost": cost}

    def _get_exploit_def(self, service, os):
        """Check if exploit parameters are valid """
        e_map = self.scenario.exploit_map
        if service not in e_map:
            return None
        if os not in e_map[service]:
            return None
        return e_map[service][os]

    def _get_privesc_def(self, proc, os):
        """Check if privilege escalation parameters are valid """
        pe_map = self.scenario.privesc_map
        if proc not in pe_map:
            return None
        if os not in pe_map[proc]:
            return None
        return pe_map[proc][os]

class ParameterisedDefenderActionSpace(spaces.MultiDiscrete):
    action_types = [
        Change_Firewall,
        Change_Os,
        Stop_Process,
        Stop_Service
    ]

    def __init__(self, scenario):
        """
        Parameters
        ----------
        scenario : Scenario
            scenario description
        """
        self.scenario = scenario
        self.actions = load_action_list(scenario)

        nvec = [
            len(self.action_types),
            len(self.scenario.subnets)-1,
            max(self.scenario.subnets),
            self.scenario.num_os+1,
            self.scenario.num_services,
            self.scenario.num_processes
        ]

        super().__init__(nvec)

    def get_action(self, action_vec):
        assert isinstance(action_vec, (list, tuple, np.ndarray)), \
            ("When using parameterised action space, action must be an Action"
             f" object, a list or a numpy array: {action_vec} is invalid")
        a_class = self.action_types[action_vec[0]]
        # need to add one to subnet to account for Internet subnet
        subnet = action_vec[1]+1
        host = action_vec[2] % self.scenario.subnets[subnet]

        target = (subnet, host)

        if a_class not in (Exploit, PrivilegeEscalation):
            # can ignore other action parameters
            kwargs = self._get_scan_action_def(a_class)
            return a_class(target=target, **kwargs)

        os = None if action_vec[3] == 0 else self.scenario.os[action_vec[3]-1]

        if a_class == Exploit:
            # have to make sure it is valid choice
            # and also get constant params (name, cost, prob, access)
            service = self.scenario.services[action_vec[4]]
            a_def = self._get_exploit_def(service, os)
        else:
            # privilege escalation
            # have to make sure it is valid choice
            # and also get constant params (name, cost, prob, access)
            proc = self.scenario.processes[action_vec[5]]
            a_def = self._get_privesc_def(proc, os)

        if a_def is None:
            return NoOp()
        return a_class(target=target, **a_def)

    def _get_scan_action_def(self, a_class):
        """Get the constants for scan actions definitions """
        if a_class == ServiceScan:
            cost = self.scenario.service_scan_cost
        elif a_class == OSScan:
            cost = self.scenario.os_scan_cost
        elif a_class == SubnetScan:
            cost = self.scenario.subnet_scan_cost
        elif a_class == ProcessScan:
            cost = self.scenario.process_scan_cost
        else:
            raise TypeError(f"Not implemented for Action class {a_class}")
        return {"cost": cost}

    def _get_exploit_def(self, service, os):
        """Check if exploit parameters are valid """
        e_map = self.scenario.exploit_map
        if service not in e_map:
            return None
        if os not in e_map[service]:
            return None
        return e_map[service][os]

    def _get_privesc_def(self, proc, os):
        """Check if privilege escalation parameters are valid """
        pe_map = self.scenario.privesc_map
        if proc not in pe_map:
            return None
        if os not in pe_map[proc]:
            return None
        return pe_map[proc][os]
